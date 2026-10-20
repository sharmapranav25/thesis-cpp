"""Record human-visible inspection; does not alter the frozen controller or scores."""
import common as U
from run import report
out=U.ROOT/'run_03'
result=U.read(out/'results.json')
review=dict(scope='Assistant inspected the unselected first 16 held-out target/start/result/mask rows in comparison.jpg.',
    improvements=['White/gray background and dark/gray text replace the initial unrelated saturated palettes.',
                  'Internal masks visibly follow most target glyph strokes in these examples.',
                  'Rendered text and background luminance are closer to the targets.'],
    failures=['Selected Mansalva handwriting font does not reproduce the target glyph shapes.',
              'Background retains or adds thin curves and scribbles unlike the target bands.',
              'Glyph height, spacing, vertical displacement and pixel character differ visibly.'],
    conclusion='Working pixels-only controller MVP with visible palette improvement in both text and background. Close family reconstruction is not achieved.',
    real_mask_accuracy='Not measured; no target masks or character labels available.',
    policy_changed_after_test=False)
result['visual_review']=review
result['conclusion']=review['conclusion']
result['decent_family_reconstruction']=False
result['distance_reduction_percent']={k:100*(1-result['test']['refined'][k]/result['test']['baseline'][k]) for k in ['text','background','appearance','audit']}
U.dump(out/'visual_review.json',review);U.dump(out/'results.json',result)
report(out,result)
state=U.read(out/'status.json');state['results']=result;U.dump(out/'status.json',state)
print(__import__('json').dumps(result['distance_reduction_percent']))
