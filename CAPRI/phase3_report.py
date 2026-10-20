"""Read-only experiment reporting; finalization requires an audited targeted dataset."""
from pathlib import Path
import argparse,json,sys
BASE=Path(__file__).resolve().parent
PROJECT=Path('E:/pranav/Final sem/sourcefree_synth')
sys.path.insert(0,str(PROJECT/'style_pipeline/phase3_reader_v2'))
from common import read,write,sha

def main(a):
    a.out.mkdir(parents=True,exist_ok=True)
    text=['# CAPRI official Phase3 — City results','', 'Synthetic-only reader comparison, frozen real-fit scheme estimation, and targeted generation. Historical optional image refiners are excluded. This report does not claim real transcription accuracy or unseen-data generalization.','', '| Reader | State | Best selection CER | Selection exact | Selected epoch |','|---|---|---:|---:|---:|']
    histories={};results={}
    for name in ['crnn','svtr','parseq']:
        for suffix in ['plain','aux']:
            tag=name+'_'+suffix;run=a.root/tag;status=read(run/'status.json') if (run/'status.json').exists() else dict(state='not_started')
            history=read(run/'history.json') if (run/'history.json').exists() else []
            if history:
                histories[tag]=history;b=min(history,key=lambda r:(r['selection']['cer'],-r['selection']['exact_string_accuracy']));text.append(f"| {tag} | {status['state']} | {b['selection']['cer']:.2%} | {b['selection']['exact_string_accuracy']:.2%} | {b['epoch']} |")
            else:text.append(f"| {tag} | {status['state']} | — | — | — |")
            if (run/'synthetic_results.json').exists():results[tag]=read(run/'synthetic_results.json')
    if histories:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        for tag,hist in histories.items():
            ax[0].plot([r['epoch'] for r in hist],[r['selection']['cer']*100 for r in hist],label=tag)
            ax[1].plot([r['epoch'] for r in hist],[r['selection']['exact_string_accuracy']*100 for r in hist],label=tag)
        for x in ax:x.set_xlabel('Epoch');x.grid(alpha=.2)
        ax[0].set_ylabel('Synthetic selection CER (%)');ax[1].set_ylabel('Synthetic selection exact text (%)');ax[1].legend(fontsize=8);fig.suptitle('City synthetic-only comparison — matched image/update budgets');fig.savefig(a.out/'learning_curves.png',dpi=160);plt.close(fig)
        text+=['','![Synthetic selection curves](learning_curves.png)','']
    if results:
        text+=['| Reader / test | CER | Exact text | Decoded count | Count head |','|---|---:|---:|---:|---:|']
        for tag,tests in results.items():
            for test,m in tests.items():
                head=f"{m['head_count_accuracy']:.2%}" if 'head_count_accuracy' in m else '—'
                text.append(f"| {tag} / {test} | {m['cer']:.2%} | {m['exact_string_accuracy']:.2%} | {m['decoded_count_accuracy']:.2%} | {head} |")
        text+=['','Per-length and digit/lowercase/uppercase precision, recall, F1 and multilabel accuracy are preserved in each synthetic_results.json. Layout test compresses the entire image horizontally and pads white; it is a stress test, not an independently rendered shortcut-free corpus. Heldout font records may mix heldout and seen fonts; no heldout-font records enter fitting or selection.','']
    if (a.root/'inferred_scheme.json').exists():
        scheme=read(a.root/'inferred_scheme.json');text+=['## Frozen fit-only estimates','',f"Scheme status: {scheme['status']}. Fit images: 1,599. Validation predictions were not opened by the aggregator. Prior mass retained for uncertainty: {scheme['uncertainty_prior_mass']:.2%}.",'', 'Bootstrap intervals reflect image sampling variability conditional on these models; they do not bound model bias or real error. Agreement and high confidence alone do not establish correctness.','']
        for tag,s in scheme['model_summaries'].items():text.append(f"- {tag}: predicted lengths {s['length_histogram']}; digit/lower/upper prevalence {s['group_prevalence']}.")
        text+=['',f"Disagreement: {scheme['disagreement']}. See auxiliary_crosschecks in inferred_scheme.json for sequence/count-head confusion and group-head disagreement.",'','All62 characters are retained. Sampling remains a weighted2–8 length range with nonzero feasible group combinations. This is an uncertainty-preserving engineering estimate, not a verified City rule. Decoders: PARSeq has a generic maximum12; CTC decodes its natural frame sequence. Count heads were trained only on2–8 and cannot establish exclusion of other real lengths.','']
    text+=['## Provenance and limits','','The original 100k corpus and all historical datasets/vendor files remain intact. Full independent audit rehashed100k unique synthetic images and1599fit/400validation real images. Real split files and pixels matched; no cross-boundary duplicates. The real directory has2000PNG images, no annotation sidecar; the manifest selects1999. No filename-derived labels are used. Independent real transcription ground truth is unavailable, so no CER or accuracy is reported for real images. Historical split exposure prevents a final generalization claim; separate unexposed data would be required.','','The initial joint span/gap metadata probe was algebraically redundant: span / mean gap +1 equals count. Its100% score alone is not independent evidence. Follow-up single-cue probes: normalized center gap96.11%, mean glyph width50.90%, mean height59.96%, versus17.23% majority baseline. These use known synthetic glyph geometry, not image detections. No shortcut-resistant counting claim. Frozen CRNN layout follow-up on1024 previously exposed synthetic test images: ordinary CER7.60%/count100%; width shrink plus white padding CER64.68%/count3.32%; replicated-edge padding CER64.64%/count3.32%; resolution reduction with original layout restored CER11.99%/count99.71%. This supports layout sensitivity; it does not prove a unique internal mechanism. These follow-up tests are development diagnostics.','','Auxiliary improvement is a matched joint-head hypothesis test, not proof of counting mechanism. One seed and different parameter/FLOP counts limit model rankings. DarkBay changed fonts, length and training duration together and cannot isolate font diversity. Raw confidence definitions differ: CRNN includes all CTC frames/blanks, SVTR uses emitted nonblank tokens, and PARSeq includes tokens/EOS. Do not compare their raw confidence values as calibrated exact-string probabilities. Synthetic reliability bins are separate; no real calibration claimed.','','Failures preserved: run01 first full batch failed on nonfinite float16 scaled gradients. Run02 uses bfloat16 without scaling; full-batch tests passed. Initial small test needed Reader.to(cuda) to propagate nested Lightning device metadata; local import precedence was also corrected. First layout diagnostic had an invalid replicate-padding tuple; partial01 and corrected02 retained. No failed historical optional refiner was reintroduced.','']
    if a.final:
        assert a.dataset is not None
        audit=read(a.dataset/'audit.json');assert audit['passed'] and read(a.dataset/'status.json')['state']=='complete'
        assert sha(a.dataset/'inferred_scheme.json')==sha(a.root/'inferred_scheme.json')
        assert len(results)==6 and all(read(a.root/tag/'status.json')['state']=='complete' for tag in results)
        upstream=Path('E:/pranav/Final sem/Models/level 1/upstream');provenance=read(a.root/'upstream_provenance.json')
        for rel,digest in {**provenance['source_hashes'],**provenance['license_hashes']}.items():assert sha(upstream/rel)==digest,rel
        source_audit=read(a.root/'data/audit.json')
        assert sha(PROJECT/'runs/pixel_statistics_01/city/split.json')==source_audit['split_sha256']
        text+=['## Targeted dataset handoff','',f"Completed dataset: `{a.dataset}`. {audit['images']:,} labelled images, {audit['unique_pixels']:,} unique pixels, all35 fonts and62 characters; minimum per-glyph visibility {audit['minimum_glyph_visibility']:.3f}. Labels, per-image recipes, hashes, model/data provenance, inferred scheme, uncertainty and sampling statistics are retained alongside the integrity audit. Final production recognizer training is a subsequent task.",'']
        write(a.root/'status.json',dict(state='complete',dataset=str(a.dataset),dataset_audit_sha256=sha(a.dataset/'audit.json'),scheme_sha256=sha(a.root/'inferred_scheme.json'),production_recognizer_trained=False))
        summary=f"\n\nOfficial CAPRI Phase3 City completed: {a.root}; targeted dataset {a.dataset}, audit passed for{audit['images']}images. See CAPRI_PHASE3_CITY_RESULTS.md. Synthetic-only six-model comparison, fit-only scheme with uncertainty, no real accuracy/generalization claim; count-layout shortcut unresolved and disclosed. Failedfloat16 run01 preserved; bf16run02 complete. Final production recognizer not trained. Monitor actual state complete; research automation remains paused.\n"
        for name in ['STYLE_HANDOFF.md','STYLE_DECISIONS.md','THESIS_PAPERS.md']:
            with (PROJECT/'docs'/name).open('a',encoding='utf8') as f:f.write(summary)
    (a.out/'CAPRI_PHASE3_CITY_RESULTS.md').write_text('\n'.join(text),encoding='utf8')
    print(a.out/'CAPRI_PHASE3_CITY_RESULTS.md')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--dataset',type=Path);p.add_argument('--final',action='store_true');main(p.parse_args())
