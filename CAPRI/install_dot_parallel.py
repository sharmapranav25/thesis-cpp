from pathlib import Path
import json,shutil,subprocess,sys
ROOT=Path('E:/pranav/Final sem/sourcefree_synth');HERE=Path(__file__).parent;src=HERE/'dot_specialist';dst=ROOT/'feature_specialists_v2/dots'
assert not dst.exists(),'Refusing to overwrite any existing dot specialist'
dst.mkdir()
for name in ('PLAN.md','corpus.py','train_dots.py','test_dots.py'):shutil.copy2(src/name,dst/name)
shutil.copy2(src/'specialists.html',ROOT/'monitor/specialists.html')
index=ROOT/'monitor/index.html';html=index.read_text(encoding='utf8')
if 'specialists.html' not in html:
    (ROOT/'monitor/index.before_parallel_specialists.html').write_text(html,encoding='utf8')
    html=html.replace('</body>','<p><a href="specialists.html">View all independent feature-specialist jobs</a></p></body>');index.write_text(html,encoding='utf8')
(ROOT/'docs/V2_DOT_SPECIALIST_PLAN.md').write_text((src/'PLAN.md').read_text(),encoding='utf8')
msg='D085: User authorizes concurrent independent specialist training. SampledRTXA4000headroom16GBtotal/1.6GBused/37%GPU. Add dot worker alongside existing curve PID26316; own20kpairedcorpus,seeds91m..97m,pop(top/bottom-hat),models,viewsearch,calibration/lockedtests. Dot means non-text roundscatter radius1..3, not arbitrarygrain. Tests4/4pass; fullcorpusgate runs beforetraining. Use monitor/specialists.html for per-job states; legacy singleton/status and old completion prose do not describe all workers. Initial concurrency capped2; no sharedweights/no curve modifications/no productionpromotion. Other specialists queued, automationpaused.'
for name in ('STYLE_DECISIONS.md','STYLE_HANDOFF.md'):
    with (ROOT/'docs'/name).open('a',encoding='utf8') as f:f.write('\n\n'+msg+'\n')
with (ROOT/'docs/THESIS_PAPERS.md').open('a',encoding='utf8') as f:f.write('\n\nIndependent dot specialist: original synthetic paired data and standard OpenCV top-hat/black-hat morphological operations as a candidate pop. No new external paper directly consulted or reproduction claimed. Source and limitations V2_DOT_SPECIALIST_PLAN.md; training/evaluation pending.\n')
run=dst/'runs/dots_v1';run.mkdir(parents=True);stdout=(run/'training_01.stdout.log').open('w');stderr=(run/'training_01.stderr.log').open('w')
# No visible shell/window; one dedicated process, with its own exclusive run lock.
process=subprocess.Popen(['E:/pranav/model 1/.venv/Scripts/python.exe',str(dst/'train_dots.py'),'--monitor'],cwd=dst,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
(run/'launch.json').write_text(json.dumps(dict(launcher_pid=process.pid,command='train_dots.py --monitor',feature='dot_scatter'),indent=2));print(json.dumps({'dot_launcher_pid':process.pid,'dashboard':'http://127.0.0.1:8765/monitor/specialists.html','run':str(run)}))
