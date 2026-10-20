from pathlib import Path
import shutil
root=Path('E:/pranav/Final sem/sourcefree_synth');here=Path(__file__).parent;source=here/'real_dot_review_01';target=root/'feature_specialists_v2/dots/reviews/real_review_01'
assert not target.exists()
shutil.copytree(source,target)
shutil.copy2(here/'review_real_dots.py',target/'review_real_dots.py')
shutil.copy2(source/'REVIEW.md',root/'docs/V2_DOT_REAL_REVIEW_01.md')
for name in ['STYLE_DECISIONS.md','STYLE_HANDOFF.md']:
 with (root/'docs'/name).open('a',encoding='utf8') as f:f.write('\n\nDot real review01: user-authorized assistant visual review of30 fit/development images (six each of five families), using frozen screening5x3pop checkpoint. City2/6 clear false positives; Courier2/6 likely texture false positives; Darkbay/outlined0/6 each; CryptBB definition ambiguous. No clear positive cohort, no recall claim, no real fitting or production promotion. Final training continues independently. See V2_DOT_REAL_REVIEW_01.md.\n')
print('Saved real review, evidence and decision log.')
