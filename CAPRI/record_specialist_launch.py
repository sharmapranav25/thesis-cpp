from pathlib import Path
import json
root=Path('E:/pranav/Final sem/sourcefree_synth');run=root/'feature_specialists_v2/runs/curves_v1';s=json.loads((run/'status.json').read_text())
message='D084 launch verified: finite independent curve-specialist worker PID '+str(s['pid'])+' active; latest observed '+s['stage']+' '+s.get('candidate','')+'. Read feature_specialists_v2/runs/curves_v1/status.json/events.jsonl and monitor for live state. Whole-image raw/pop screening near chance; no learning-success claim. View search ongoing; do not duplicate worker. No real fitting, no pipeline promotion, other features queued, continuous automation still paused. Corpus20000images includes12000training images; remaining8000selection/calibration/locked evaluation.'
for name in ('STYLE_HANDOFF.md','STYLE_DECISIONS.md'):
    with (root/'docs'/name).open('a',encoding='utf8') as f:f.write('\n\n'+message+'\n')
print(message)
