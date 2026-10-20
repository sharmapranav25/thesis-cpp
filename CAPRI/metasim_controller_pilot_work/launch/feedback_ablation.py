"""Equal-budget City controller searches with either DINO or statistics feedback."""
import argparse
import copy
import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import pilot as Q
import pilot_render as P

WEIGHTS = {'dino_only': (1.0, 0.0), 'stats_only': (0.0, 1.0), 'mixed': (.6, .4)}


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines()]


class Feedback(Q.Feedback):
    def __init__(self, fit, mode):
        super().__init__(fit)
        self.weights = WEIGHTS[mode]

    def score(self, f, real, rows):
        components = self.components(f, real)
        usable = float(np.mean([r['usable'] for r in rows]))
        strict = float(np.mean([r['strict'] for r in rows]))
        appearance = self.weights[0]*components['dino']+self.weights[1]*components['stats']
        return dict(**components, appearance=appearance,
            hybrid_reference=.6*components['dino']+.4*components['stats'],
            usable=usable, strict=strict,
            objective=appearance+.8*(1-usable)+.15*(1-strict))


def event(out, started, stage, **values):
    row=dict(stage=stage, seconds=round(time.perf_counter()-started,1), **values)
    Q.dump(out/'status.json', row)
    with (out/'events.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(row,default=Q.default)+'\n')
    print(json.dumps(row,default=Q.default),flush=True)


def prior_group(parent, name):
    rows=lines(parent/name/'labels.jsonl')
    for r in rows:
        with Image.open(parent/r['image']) as im:
            pixels=np.asarray(im.convert('RGB'))
        assert hashlib.sha256(pixels.tobytes()).hexdigest()==r['pixel_sha256']
        r['image']=pixels
    return rows,dict(np.load(parent/f'{name}_features.npz'))


def paired_intervals(feedback, before, after, real, resamples=200):
    b,a,r=map(feedback.normalize,(before,after,real))
    rng=np.random.default_rng(Q.SEED+37)
    ks={key:(Q.kernel(b[key],b[key],feedback.widths[key]),
             Q.kernel(a[key],a[key],feedback.widths[key]),
             Q.kernel(b[key],r[key],feedback.widths[key]),
             Q.kernel(a[key],r[key],feedback.widths[key])) for key in ['dino','stats']}
    delta=[]
    for _ in range(resamples):
        si=rng.integers(len(b['dino']),size=len(b['dino']))
        ri=rng.integers(len(r['dino']),size=len(r['dino']))
        changes=[]
        for key in ['dino','stats']:
            kb,ka,kbr,kar=ks[key]
            changes.append(float(ka[np.ix_(si,si)].mean()-kb[np.ix_(si,si)].mean()
                                 -2*(kar[np.ix_(si,ri)].mean()-kbr[np.ix_(si,ri)].mean())))
        delta.append(changes+[.6*changes[0]+.4*changes[1]])
    delta=np.asarray(delta)
    return dict(resamples=resamples,after_minus_specialist_before_95pct={
        name:np.quantile(delta[:,i],[.025,.975]).tolist()
        for i,name in enumerate(['dino','stats','hybrid_reference'])})


def summary(feedback, features, real, rows):
    s=feedback.score(features,real,rows)
    s['audit_sliced_wasserstein']=feedback.audit(features,real)
    s['images']=len(rows)
    s['complete_text']=sum(r['text_complete'] for r in rows)
    s['unique_pixels']=len({r.get('pixel_sha256') for r in rows})
    s['unique_strings']=len({r.get('text') for r in rows})
    s['font_categories']=dict(Counter(r.get('font',{}).get('dominant_category','render_failure') for r in rows))
    for key in ['median_weight','median_width','median_x_height','median_slant_level']:
        valid=[r['font'][key] for r in rows if r.get('font',{}).get(key) is not None]
        s[key]=float(np.mean(valid)) if valid else None
    s['italic_fraction']=float(np.mean([r.get('font',{}).get('italic_any',False) for r in rows]))
    return s


def gallery(out, real, groups):
    names=['City references','Simple start','Specialist start','60 DINO 40 stats','100 DINO','100 stats']
    sheet=Image.new('RGB',(6*300,42+12*100),'white')
    draw=ImageDraw.Draw(sheet)
    for j,name in enumerate(names): draw.text((j*300+6,12),name,fill='black')
    for i in range(12):
        for j,pixels in enumerate([real[i]]+[rows[i]['image'] for rows in groups.values()]):
            im=Image.fromarray(pixels);im.thumbnail((292,72))
            x=j*300+4;y=42+i*100
            sheet.paste(im,(x,y+(72-im.height)//2))
            text='reference (unpaired)' if j==0 else f"{i}: {list(groups.values())[j-1][i].get('text','render failure')}"
            draw.text((x,y+76),text,fill='black')
    sheet.save(out/'comparison.jpg',quality=94)
    sheet.crop((0,0,sheet.width,42+3*100)).save(out/'preview.jpg',quality=94)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();parent=args.parent.resolve();out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    previous_protocol=Q.read(parent/'protocol.json')
    immutable={path:Q.sha(parent/path) for path in ['protocol.json','results.json','selected_controls.json','target_split.json']}
    sources=previous_protocol['source_hashes']
    assert all(Q.sha(path)==value for path,value in sources.items())
    previous_environment=Q.read(parent/'environment.json')
    renderer_sources={path:sha for path,sha in previous_environment['sources'].items() if str(P.PROJECT.parent/'captcha-effects') in path or 'captcha-effects' in path}
    assert all(Q.sha(path)==value for path,value in renderer_sources.items())
    protocol=dict(family='City',parent=str(parent),modes={name:dict(dino=w[0],stats=w[1]) for name,w in WEIGHTS.items() if name!='mixed'},
        rounds=8,population=8,fit_renders_per_candidate=24,candidate_renders_per_mode=1536,
        fit_target_images=256,selection_target_images=128,selection_synthetic_images=48,
        evaluation_target_images=512,evaluation_synthetic_images_per_group=96,
        common_search_seed=Q.SEED+20,controls=dict(zip(P.NAMES,P.BOUNDS.tolist())),
        selection='Each mode chooses the best eligible of its three best-fit candidates and neutral; same 5-point guard tolerance as original',
        readability_penalties=dict(unusable=.8,strict_failure=.15),
        appearance_weights_only_change=True,neural_updates=0,real_answers_used=False,
        mode='Fresh independent searches, not rescoring the original controller',
        feature_extraction='Frozen DINO and statistics both computed for cross-metric reporting; only the named channel contributes to each mode fitness',
        evaluation='New real images excluded from all target subsets of the earlier pilot; no evaluation metric before all modes are selected',
        source_hashes=sources,renderer_hashes=renderer_sources,parent_hashes=immutable)
    Q.dump(out/'protocol.json',protocol)
    old_split=Q.read(parent/'target_split.json')['splits']
    excluded={r['pixel_sha256'] for values in old_split.values() for r in values}
    fit=old_split['fit'];selection=old_split['selection']
    allrows=lines(Q.SPECIAL/'real_inference/predictions.jsonl')
    unseen={}
    for row in allrows:
        if row['family']!='City':continue
        with Image.open(row['source']) as im:
            pixel_hash=hashlib.sha256(np.asarray(im.convert('RGB')).tobytes()).hexdigest()
        if pixel_hash not in excluded and pixel_hash not in unseen:
            unseen[pixel_hash]=dict(source=row['source'],pixel_sha256=pixel_hash)
    assert len(unseen)>=512
    unseen=list(unseen.values());order=np.random.default_rng(Q.SEED+55).permutation(len(unseen))
    test=[unseen[i] for i in order[:512]]
    Q.dump(out/'target_split.json',dict(fit=fit,selection=selection,fresh_evaluation=test,
        excluded_all_previous_splits=len(excluded),unused_available=len(unseen),remaining_unused=len(unseen)-512))
    common=Q.read(parent/'fit_recipes.json');fonts=[r['key'] for r in Q.read(parent/'font_candidates.json')]
    specialist_profile=Q.read(parent/'specialist_initial_profile/profiles/City.json')
    Q.dump(out/'fit_recipes.json',common);Q.dump(out/'font_candidates.json',Q.read(parent/'font_candidates.json'))
    event(out,started,'extracting_fit_and_selection',fresh_evaluation_images=512)
    model=Q.Features();real_fit=model(Q.real_images(fit));real_selection=model(Q.real_images(selection))
    reference=Feedback(real_fit,'mixed')
    old_scaling=np.load(parent/'feedback_scaling.npz')
    assert np.allclose(reference.mean,old_scaling['stats_mean']) and np.allclose(reference.std,old_scaling['stats_std'])
    assert all(np.isclose(reference.widths[key],value,rtol=1e-5) for key,value in Q.read(parent/'kernel_bandwidths.json').items())
    np.savez(out/'target_fit_features.npz',**real_fit)
    np.savez(out/'target_selection_features.npz',**real_selection)
    np.savez(out/'feedback_scaling.npz',stats_mean=reference.mean,stats_std=reference.std,
             audit_mean=reference.audit_mean,audit_std=reference.audit_std,directions=reference.directions)
    Q.dump(out/'kernel_bandwidths.json',reference.widths)
    selected={};histories={};mode_elapsed={};search_renders={}
    with ProcessPoolExecutor(max_workers=6,initializer=P.install) as pool:
        neutral=Q.generated(pool,common,P.NEUTRAL,fonts)
        original=P.plain(common[0]);assert np.array_equal(original['image'],neutral[0]['image'])
        neutral_features=Q.features(model,neutral)
        Q.dump(out/'neutral_replay_check.json',dict(passed=True,pixel_max_abs_difference=0))
        event(out,started,'preparing_common_selection_recipes')
        selection_recipes=Q.recipes(pool,specialist_profile,48,Q.SEED+30)
        Q.dump(out/'selection_recipes.json',selection_recipes)
        neutral_selection=Q.generated(pool,selection_recipes,P.NEUTRAL,fonts)
        neutral_selection_features=Q.features(model,neutral_selection)
        selection_check=reference.score(neutral_selection_features,real_selection,neutral_selection)
        previous_selection=Q.read(parent/'selection.json')[0]['score']
        assert all(np.isclose(selection_check[k],previous_selection[k],atol=1e-5) for k in ['dino','stats','usable','strict'])
        for mode in ['dino_only','stats_only']:
            mode_start=time.perf_counter();folder=out/mode;folder.mkdir()
            feedback=Feedback(real_fit,mode)
            initial_score=feedback.score(neutral_features,real_fit,neutral)
            event(out,started,'search_start',mode=mode,initial=initial_score)
            rng=np.random.default_rng(Q.SEED+20);mean=P.NEUTRAL.copy()
            span=P.BOUNDS[:,1]-P.BOUNDS[:,0];std=.35*span
            history=[];evaluated=0
            for iteration in range(8):
                for member in range(8):
                    if iteration==0 and member==0:vector=P.NEUTRAL.copy()
                    elif iteration==0 or rng.random()<.25:vector=rng.uniform(P.BOUNDS[:,0],P.BOUNDS[:,1])
                    else:vector=np.clip(rng.normal(mean,std),P.BOUNDS[:,0],P.BOUNDS[:,1])
                    rendered=Q.generated(pool,common,vector,fonts)
                    score=feedback.score(Q.features(model,rendered),real_fit,rendered)
                    history.append(dict(round=iteration+1,member=member,vector=vector.tolist(),**score));evaluated+=24
                    event(out,started,'searching',mode=mode,round=iteration+1,member=member+1,
                          candidate_renders=evaluated,score=score['objective'],best=min(r['objective'] for r in history))
                    Q.dump(folder/'search_history.json',history)
                elite=sorted(history,key=lambda r:r['objective'])[:4]
                values=np.array([r['vector'] for r in elite])
                mean=.5*mean+.5*values.mean(0);std=np.maximum(.12*span,.5*std+.5*values.std(0))
            best=[];seen=set()
            for c in sorted(history,key=lambda r:r['objective']):
                identity=tuple(np.round(c['vector'],5))
                if identity not in seen:best.append(c);seen.add(identity)
                if len(best)==3:break
            event(out,started,'selecting',mode=mode)
            baseline=feedback.score(neutral_selection_features,real_selection,neutral_selection)
            choices=[dict(vector=P.NEUTRAL.tolist(),score=baseline,eligible=True,source='neutral')]
            for i,candidate in enumerate(best):
                rendered=Q.generated(pool,selection_recipes,candidate['vector'],fonts)
                score=feedback.score(Q.features(model,rendered),real_selection,rendered)
                eligible=score['usable']>=baseline['usable']-.05 and score['strict']>=baseline['strict']-.05
                choices.append(dict(vector=candidate['vector'],score=score,eligible=eligible,source=f'fit_rank_{i+1}'))
            chosen=min((r for r in choices if r['eligible']),key=lambda r:r['score']['objective'])
            Q.dump(folder/'selection.json',choices)
            Q.dump(folder/'selected_controls.json',dict(choice=chosen,controls=dict(zip(P.NAMES,chosen['vector'])),frozen_before_evaluation=True))
            selected[mode]=chosen;histories[mode]=history;search_renders[mode]=evaluated
            mode_elapsed[mode]=time.perf_counter()-mode_start
            event(out,started,'mode_selected',mode=mode,choice=chosen['source'],score=chosen['score'],seconds_for_mode=mode_elapsed[mode])
        assert all(np.array_equal(histories['dino_only'][i]['vector'],histories['stats_only'][i]['vector']) for i in range(8))
        Q.dump(out/'selected_all_controls.json',dict(selected=selected,all_frozen_before_evaluation=True))
        selected_sha=Q.sha(out/'selected_all_controls.json')
        event(out,started,'fresh_evaluation',images=512)
        test_images=Q.real_images(test)
        for pixels,row in zip(test_images,test):
            assert hashlib.sha256(pixels.tobytes()).hexdigest()==row['pixel_sha256']
        real_test=model(test_images);np.savez(out/'target_evaluation_features.npz',**real_test)
        groups={};all_features={}
        for old_name,new_name in [('simple_initial','simple_initial'),('specialist_initial','specialist_initial'),('specialist_adjusted','mixed_adjusted')]:
            groups[new_name],all_features[new_name]=prior_group(parent,old_name)
        baseline=groups['specialist_initial']
        test_recipes=[]
        for row in baseline:
            h,w=row['image'].shape[:2]
            test_recipes.append(dict(index=row['index'],seed=row['seed'],size=[w,h],
                effects_spec=row['effects_spec'],text_spec=row['text_spec']))
        Q.dump(out/'evaluation_recipes.json',test_recipes)
        replay=Q.generated(pool,[test_recipes[0]],P.NEUTRAL,fonts)[0]
        assert replay['pixel_sha256']==baseline[0]['pixel_sha256']
        for mode in ['dino_only','stats_only']:
            name=mode+'_adjusted'
            groups[name]=Q.generated(pool,test_recipes,selected[mode]['vector'],fonts)
            all_features[name]=Q.features(model,groups[name])
            assert all(r.get('text')==b['text'] and r.get('seed')==b['seed'] for r,b in zip(groups[name],baseline))
        metrics={};intervals={};checks={}
        for name,rendered in groups.items():
            metrics[name]=summary(reference,all_features[name],real_test,rendered)
            labels=Q.save_group(out,name,rendered);np.savez(out/f'{name}_features.npz',**all_features[name])
            complete=all(r['text']==r['text_spec']['text'] and len(r['glyphs'])==len(r['text']) and all(g['box'] is not None for g in r['glyphs']) for r in rendered if 'text' in r)
            checks[name]=dict(images=len(rendered),complete_text_count=sum(r['text_complete'] for r in rendered),
                              label_integrity=complete,unique_pixels=metrics[name]['unique_pixels'],unique_strings=metrics[name]['unique_strings'])
            if name.endswith('adjusted'):
                intervals[name]=paired_intervals(reference,all_features['specialist_initial'],all_features[name],real_test)
        assert Q.sha(out/'selected_all_controls.json')==selected_sha
        assert all(Q.sha(parent/path)==value for path,value in immutable.items())
        assert all(Q.sha(path)==value for path,value in sources.items())
        assert all(Q.sha(path)==value for path,value in renderer_sources.items())
        gallery(out,test_images,groups)
        Q.dump(out/'verification.json',dict(fresh_evaluation_excludes_all_previous_splits=True,
            previous_sources_unchanged=True,prior_results_unchanged=True,neutral_replay_exact=True,
            same_first_round_candidates=True,all_controllers_fixed_before_evaluation=True,
            same_paired_specialist_strings_and_seeds=96,group_checks=checks))
        Q.dump(out/'results.json',dict(groups=metrics,paired_bootstrap=intervals,selected=selected,
            search_renders=search_renders,mode_seconds=mode_elapsed,total_seconds=time.perf_counter()-started,
            evaluation_target='512 new City images excluded from all previous pilot subsets',neural_updates=0,
            primary_comparison='DINO, stats, and common 60/40 reference reported for every controller; compare each column across controllers',
            limitations=['City only','One search seed per feedback choice','Appearance proxies do not establish font or OCR accuracy',
                         'Same strings and seeds, but shared renderer random streams can change other style draws',
                         'Stats-only uses DINO for cross-metric reporting, never in fitting or selection fitness']))
    event(out,started,'complete',groups=metrics,search_renders=search_renders,neural_updates=0)


if __name__=='__main__':main()
