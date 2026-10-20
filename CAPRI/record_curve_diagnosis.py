from pathlib import Path
import shutil
root=Path('E:/pranav/Final sem/sourcefree_synth')
here=Path(__file__).parent
target=root/'feature_specialists_v2/diagnostics/curve_v1_failure'
target.mkdir(parents=True,exist_ok=True)
for name in ('diagnose_curves.py','curve_diagnosis.json','CURVE_FAILURE_DIAGNOSIS.md'):
    shutil.copy2(here/name,target/name)
shutil.copy2(here/'CURVE_FAILURE_DIAGNOSIS.md',root/'docs/V2_CURVE_FAILURE_DIAGNOSIS.md')
note='\n\nCurve v1 investigation: near-constant training predictions (~0.496; AP0.513 on 512 training examples), despite successful 16-image memorization control. Calibration tie chose0.99, explaining zero recall but not chance learning. See V2_CURVE_FAILURE_DIAGNOSIS.md for confirmed findings versus unproven training-design hypotheses. No production promotion; curve worker completed, dot worker remains independent. Monitor specialist status endpoints verified HTTP200 after narrow server allowlist fix. Further curriculum/localization ablations proposed, not yet run.\n'
for name in ('STYLE_DECISIONS.md','STYLE_HANDOFF.md'):
    with (root/'docs'/name).open('a',encoding='utf8') as f:f.write(note)
print('Recorded curve diagnosis; historical sources and checkpoints unchanged.')
