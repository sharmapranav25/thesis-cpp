import copy,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from feature_bank.iter2 import balanced as B

if __name__=='__main__':
    failed=json.loads(Path(r'E:\pranav\Final sem\sourcefree_synth\runs\iter2_baseline_15_20261003_03\train_080000_failed_tickets.json').read_text())[0]['ticket']
    scheduled,_=B.schedule(100000,2026100307,font_split='train',split='train')
    donor=next(t for t in scheduled[80000:90000] if t['k']==5 and t['appearance']=='neutral' and t['category']=='Sans Serif' and not t['italic'] and t['weight']>=4 and t['width']>=4)
    jobs=[]
    for mode,extra in [('profile',[]),('profile_weight',['weight']),('profile_width',['width']),('profile_both',['weight','width'])]:
        left,right=copy.deepcopy(failed),copy.deepcopy(donor)
        for key in ['effects','category','x_height','italic']+extra:
            left[key],right[key]=right[key],left[key]
        for ticket in [left,right]:
            ticket.update(revision=70000,max_attempts=60)
        jobs.extend([left,right])
    print(json.dumps({'donor':donor}),flush=True)
    with ProcessPoolExecutor(8) as pool:
        for i,row in enumerate(pool.map(B.render_ticket,jobs)):
            print(json.dumps({'mode':i//2,'side':i%2,'failed':row['failed'],'rejections':row['rejections'],'quality':row.get('label',{}).get('quality')}),flush=True)
