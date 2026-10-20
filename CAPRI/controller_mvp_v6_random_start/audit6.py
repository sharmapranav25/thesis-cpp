"""Check seed independence, equal budgets, target disjointness and immutable selections."""
from ablate6 import *
import ast,hashlib

def main():
    fonts=U.read(WORK/'learned_controller_20261004/fonts.json');summary=U.read(ROOT/'summary.json')
    protocol=U.read(ROOT/'protocol.json');assert protocol['code_sha256']==U.sha(ROOT/'ablate6.py')
    assert protocol['weights_sha256']==U.sha(WEIGHTS_PATH)
    prior=ROOT.parent/'controller_mvp_v5'
    for name,digest in U.read(prior/'completion.json')['source_sha256'].items():
        assert U.sha(prior/name)==digest,'Previous controller code was changed'
    details={}
    for family,result in summary.items():
        dest=ROOT/family;split=U.read(dest/'target_splits.json')
        hashes=[{r['pixel_sha256'] for r in split[k]} for k in COUNTS]
        assert all(not hashes[i]&hashes[j] for i in range(3) for j in range(i))
        shared=None;details[family]={}
        for arm,run in result['runs'].items():
            folder=dest/arm;bank=U.read(folder/'recipes.json');records=U.read(folder/'history.json')
            assert len(records)==192 and len(U.read(folder/'selection.json'))==16
            assert U.sha(folder/'selected.json')==run['selected_sha256']
            seeds={p:[r['seed'] for r in recipes] for p,recipes in bank.items()}
            if shared is None:shared=seeds
            else:assert seeds==shared
            if arm.startswith('random_'):
                seed=int(arm.split('_')[1]);size=bank['fit'][0]['size']
                for (phase,n),offset in zip(COUNTS.items(),[0,100,200]):
                    expected=random_recipes(seed,n,size,fonts,offset)
                    assert key(expected)==key(bank[phase])
                    assert all(set(r['description'])=={'content'} for r in expected)
            for phase in ['start','final']:
                labels=U.read(folder/phase/'labels.json');assert len(labels)==24
                for row in labels:
                    im=np.asarray(__import__('PIL.Image',fromlist=['Image']).open(folder/row['image']).convert('RGB'))
                    if 'pixel_sha256' in row:assert hashlib.sha256(im.tobytes()).hexdigest()==row['pixel_sha256']
            details[family][arm]=dict(fit_slots=len(records),selection_slots=16,unique_fit_controls=run['unique_fit_controls'],
                render_errors=sum(r['render_errors'] for r in records),search_pass=run['search_pass'])
    # Prove prior function has no hybrid mapping/initialization calls or file reads.
    tree=ast.parse((ROOT/'ablate6.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='random_recipes')
    text=ast.get_source_segment((ROOT/'ablate6.py').read_text(),fn)
    assert not any(term in text for term in ['initial_recipe(', 'map_recipe(', 'U.read(', 'prepare(', 'HYBRID'])
    U.dump(ROOT/'audit.json',dict(passed=True,families=len(summary),runs=sum(len(r['runs']) for r in summary.values()),
        generated_images=sum(len(r['runs'])*48 for r in summary.values()),random_prior_reproduced=True,
        shared_seeds=True,equal_budgets=True,previous_controller_unchanged=True,disjoint_pixel_splits=True,selections_unchanged=True,details=details))
    print('Audit passed:',len(summary),'families')

if __name__=='__main__':main()
