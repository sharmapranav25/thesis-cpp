import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import balanced_fast as B
import corpus_fast as C

if __name__=='__main__':
    C.B=B
    root=Path(__file__).parent/'speed_2000_01'
    root.mkdir(exist_ok=False)
    tickets,heads=B.schedule(2000,2026100302,font_split='train',split='steering')
    margins={key:Counter(t[key] for t in tickets) for key in ['length','category','content','appearance','weight','width','italic','x_height','k']}
    effects=Counter(e for t in tickets for e in t['effects'])
    start=time.perf_counter()
    with ProcessPoolExecutor(20) as pool:
        report=C.make_part(pool,root,tickets,'steering')
    result=dict(seconds=time.perf_counter()-start,reference_seconds=255,
                accepted=report['accepted'],duplicates=report['duplicates'],target_pm1=report['target_pm1'])
    result['speedup']=255/result['seconds']
    for key,value in margins.items():
        assert value==Counter(t[key] for t in tickets),key
    assert effects==Counter(e for t in tickets for e in t['effects'])
    assert report['accepted']==2000 and report['duplicates']==0
    assert report['target_pm1']['weight']>=.8
    assert report['target_pm1']['x_height']==1
    (root/'speed_result.json').write_text(json.dumps(result,indent=2))
    print('RESULT '+json.dumps(result),flush=True)
