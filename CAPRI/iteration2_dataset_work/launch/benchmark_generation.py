import copy
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import balanced_reference as Reference
import balanced_fast as Fast
import corpus as OldCorpus
import corpus_fast as FastCorpus

if __name__ == '__main__':
    base=Path(__file__).parent/'speed_benchmark_01'
    base.mkdir(exist_ok=False)
    planned,_=Reference.schedule(420,2026100302,split='steering')
    results={}
    for name, sampler, pipeline in [('reference',Reference,OldCorpus),('fast',Fast,FastCorpus)]:
        root=base/name
        root.mkdir()
        pipeline.B=sampler
        start=time.perf_counter()
        with ProcessPoolExecutor(20) as pool:
            report=pipeline.make_part(pool,root,copy.deepcopy(planned),'steering')
        results[name]=dict(seconds=time.perf_counter()-start,accepted=report['accepted'],
                           target_pm1=report['target_pm1'],duplicates=report['duplicates'],
                           font_levels=report['font_levels'],effect_levels=report['effect_levels'])
        print('BENCHMARK '+json.dumps(dict(name=name,**{k:v for k,v in results[name].items() if k not in ['font_levels','effect_levels']})),flush=True)
    results['speedup']=results['reference']['seconds']/results['fast']['seconds']
    (base/'comparison.json').write_text(json.dumps(results,indent=2))
    assert results['fast']['accepted']==420 and results['fast']['duplicates']==0
    assert results['fast']['target_pm1']['weight'] >= .8
    assert results['fast']['target_pm1']['x_height']==1
    print('SPEEDUP '+str(results['speedup']),flush=True)
