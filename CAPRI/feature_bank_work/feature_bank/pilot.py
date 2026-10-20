"""Bounded 64-case pilots; errors and repeatability failures remain visible."""
import argparse
import copy
import json
import shutil
import string
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .core import digest, read, write, load, pixel_hash, new_run
from .resources import inventory

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = {'gregwar':'Gregwar/Captcha','securimage':'dapphp/securimage',
            'mobicms':'mobicms/captcha','simple':'yasirmturk/simple-php-captcha'}
MODES = {'gregwar':['clean','lines','warp','lines_warp'],
         'securimage':['clean','curves','dots','curves_dots_warp'],
         'mobicms':['single_font','mixed_fonts','single_font','mixed_fonts'],
         'simple':['clean','shadow','rotation','shadow_rotation'],
         'python':['clean','lines','dots','blocks_grid_warp']}

def gallery(rows, out, backend):
    for page in range(2):
        subset=rows[page*32:(page+1)*32]
        canvas=Image.new('RGB',(1320,40+8*155),'#eeeeee'); draw=ImageDraw.Draw(canvas)
        draw.text((10,10),f'{backend} pilot {page+1}/2 | 64 attempts, failures retained',fill='black')
        for i,row in enumerate(subset):
            x=10+(i%4)*330; y=40+(i//4)*155
            draw.text((x,y),f"{row['id']} {row['mode']}",fill='black')
            path=out/row['image']
            if path.exists():
                im=load(path); im.thumbnail((310,112)); canvas.paste(im,(x,y+20))
            else:
                draw.text((x,y+30),'RENDER FAILED - see results.json',fill='red')
            status=row['status']
            if row.get('repeat_equal') is False: status+=' / REPLAY FAIL'
            if row.get('text_touches_edge'): status+=' / TEXT EDGE'
            draw.text((x,y+134),status,fill='black')
        canvas.save(out/f'{backend}_gallery_{page+1}.png')

def php_command(resources):
    php=resources/'php'
    return [str(php/'php.exe'),'-n','-d',f'extension_dir={php/"ext"}',
            '-d','extension=gd','-d','extension=mbstring','-d','extension=fileinfo',
            '-d','display_errors=stderr',str(ROOT/'adapters/php_pilot.php')]

def execute_php(command,cfg,folder):
    write(folder/'config.json',cfg)
    result=subprocess.run(command+[str(folder/'config.json')],capture_output=True,text=True,timeout=40)
    (folder/'stderr.txt').write_text(result.stderr,encoding='utf-8')
    (folder/'stdout.txt').write_text(result.stdout,encoding='utf-8')
    if result.returncode:
        raise RuntimeError(result.stderr[:1600])
    im=load(cfg['output'])
    metadata=read(cfg['output']+'.json')
    if metadata['actual']['label']!=cfg['label']:
        raise ValueError('Exported/internal text mismatch')
    if im.size!=(cfg['width'],cfg['height']):
        raise ValueError('Unexpected dimensions')
    if np.asarray(im).std()<1:
        raise ValueError('Blank/degenerate output')
    return im,metadata

def run(resources, project, output):
    resources,project,out=Path(resources).resolve(),Path(project).resolve(),new_run(output).resolve()
    write(out/'status.json',{'stage':2,'state':'RUNNING'})
    sources,assets=inventory(resources,out,project)
    fonts=[a for a in assets if a['license_status']=='OFL-1.1_notice_present' and not a['missing_ascii_alphanumeric']]
    if len(fonts)<2: raise ValueError('Need at least two licensed generic fonts covering ASCII')
    # Restrict MobiCMS to vetted fonts; retain license notices alongside copies.
    font_dir=out/'generic_fonts'; font_dir.mkdir()
    for a in fonts[:3]:
        shutil.copy2(a['path'],font_dir/Path(a['path']).name)
        shutil.copy2(a['license_path'],font_dir/Path(a['license_path']).name)
    write(out/'font_selection.json',fonts[:3])
    command=php_command(resources)
    results={}; start=time.perf_counter()
    rng=np.random.default_rng(20260928)
    labels=[''.join(rng.choice(list(string.ascii_letters+string.digits),size=3+i%6)) for i in range(16)]
    # Include hard-negative glyphs without reading or imitating target labels.
    labels[:3]=['Il1','OO0Q','ijIl']
    for backend,repo in BACKENDS.items():
        rows=[]
        for i in range(64):
            folder=out/backend/f'{i:03d}'; folder.mkdir(parents=True)
            trace=folder/'trace'; trace.mkdir()
            mode=MODES[backend][i%4]; w,h=[(200,80),(240,96),(320,120)][(i//4)%3]
            font=fonts[(i//4)%min(3,len(fonts))]
            background=folder/'background.png'; Image.new('RGB',(w,h),(245,245,245)).save(background)
            cfg=dict(backend=backend,source=sources[repo],font=font['path'],font_dir=str(font_dir),
               label=labels[i//4],seed=1000+i//4,width=w,height=h,mode=mode,
               output=str(folder/'image.png'),trace_dir=str(trace),background=str(background),font_size=18)
            row=dict(id=f'{backend}-{i:03d}',base_group=f'{backend}-{i//4:02d}',mode=mode,
                     image=f'{backend}/{i:03d}/image.png',requested=cfg,training_eligible=False)
            try:
                im,meta=execute_php(command,cfg,folder)
                row.update(status='RENDERED',pixel_sha256=pixel_hash(im),actual=meta['actual'],warnings=meta['warnings'])
                repeat=folder/'repeat'; repeat.mkdir(); (repeat/'trace').mkdir()
                cfg2=dict(cfg,output=str(repeat/'image.png'),trace_dir=str(repeat/'trace'))
                im2,_=execute_php(command,cfg2,repeat)
                row['repeat_equal']=pixel_hash(im)==pixel_hash(im2)
                before,after=trace/'01_before_text.png',trace/'02_after_text.png'
                if before.exists() and after.exists():
                    delta=np.any(np.asarray(load(before))!=np.asarray(load(after)),axis=2)
                    Image.fromarray((delta*255).astype('uint8')).save(trace/'text_change_mask.png')
                    row['text_touches_edge']=bool(delta[0].any() or delta[-1].any() or delta[:,0].any() or delta[:,-1].any())
                    row['mask_limitation']='before-warp text change, not final per-effect ground truth'
                if backend=='simple':
                    row['bbox_outside_canvas']=any(x<0 or x>=w or y<0 or y>=h for x,y in meta['actual']['text_bbox_corners'])
            except Exception as exc:
                row.update(status='FAILED',error=str(exc))
            rows.append(row)
            write(out/backend/'results.json',rows)
        results[backend]=rows; gallery(rows,out,backend)
        print(backend,summary(rows),flush=True)
    # Reuse the independent Python renderer unchanged with fixed generic priors.
    sys.path.insert(0,str(project))
    from sourcefree_synth.renderer import render_once,default_parameters
    from sourcefree_synth.data import sha256_file
    rows=[]
    for i in range(64):
        folder=out/'python'/f'{i:03d}'; folder.mkdir(parents=True)
        mode=MODES['python'][i%4]; w,h=[(200,80),(240,96),(320,120)][(i//4)%3]
        params=default_parameters(0)
        params.update(background='flat',dots=0,lines=0,rotation=0,jitter=0,stroke=0,
                      outline=False,shadow=False,multicolor=False,wave_x=0,wave_y=0,blur=0)
        if mode=='lines': params['lines']=3
        if mode=='dots': params['dots']=0.006
        if mode=='blocks_grid_warp': params.update(background='blocks_grid',wave_y=0.025,lines=1,dots=0.003)
        profile=dict(id='generic-fixed-pilot-v1',width=w,height=h,background=[[245,245,245],[180,210,225]],foreground=[[30,40,65]])
        font=fonts[(i//4)%min(3,len(fonts))]; trace={}
        row=dict(id=f'python-{i:03d}',base_group=f'python-{i//4:02d}',mode=mode,image=f'python/{i:03d}/image.png',training_eligible=False,
                 requested=dict(label=labels[i//4],parameters=params,profile=profile,font=font,seed=1000+i//4))
        try:
            im,mask,_,meta=render_once(labels[i//4],params,profile,[font],1000+i//4,trace)
            im.save(folder/'image.png'); mask.save(folder/'text_mask.png')
            im2,_,_,_=render_once(labels[i//4],params,profile,[font],1000+i//4)
            row.update(status='RENDERED',actual=meta,pixel_sha256=pixel_hash(im),repeat_equal=pixel_hash(im)==pixel_hash(im2))
        except Exception as exc:
            row.update(status='FAILED',error=str(exc))
        for name,im in trace.items(): im.save(folder/(name+'.png'))
        rows.append(row)
    write(out/'python/results.json',rows); results['python']=rows; gallery(rows,out,'python')
    print('python',summary(rows),flush=True)
    # Deliberately invalid requests must fail before renderer invocation.
    tests=[]; example=copy.deepcopy(results['gregwar'][0]['requested'])
    for key,value in [('width',-1),('label','bad!'),('mode','unknown'),('font','missing-font.ttf'),('invented_parameter',1)]:
        folder=out/'invalid'/key; folder.mkdir(parents=True)
        cfg=dict(example); cfg[key]=value; cfg['output']=str(folder/'must_not_exist.png')
        write(folder/'config.json',cfg)
        p=subprocess.run(command+[str(folder/'config.json')],capture_output=True,text=True,timeout=40)
        tests.append(dict(parameter=key,rejected=p.returncode!=0 and not Path(cfg['output']).exists(),stderr=p.stderr))
    write(out/'invalid_parameter_checks.json',tests)
    inventory(resources,out,project)
    report=dict(stage=2,state='PILOTS_COMPLETE_REVIEW_PENDING',backends={k:summary(v) for k,v in results.items()},
      elapsed_seconds=time.perf_counter()-start,invalid_parameter_checks=tests,
      python_renderer_sha256=sha256_file(project/'sourcefree_synth/renderer.py'),
      code_sha256={str(p.relative_to(ROOT)):digest(p) for folder in ('feature_bank','adapters') for p in (ROOT/folder).rglob('*') if p.is_file()},
      withheld_laminas='NOT_RUN_dependency_resolution_pending',training_started=False)
    write(out/'status.json',report)
    return report

def summary(rows):
    return dict(attempted=len(rows),rendered=sum(r['status']=='RENDERED' for r in rows),
       failed=sum(r['status']=='FAILED' for r in rows),
       exact_replays=sum(r.get('repeat_equal') is True for r in rows),
       replay_mismatches=sum(r.get('repeat_equal') is False for r in rows),
       edge_or_bbox_flags=sum(bool(r.get('text_touches_edge') or r.get('bbox_outside_canvas')) for r in rows))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--resources',required=True); p.add_argument('--project',required=True); p.add_argument('--out',required=True)
    a=p.parse_args(); run(a.resources,a.project,a.out)
