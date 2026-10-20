"""Qualitative review recorded after locked evaluation; no controller changes."""
import common as U
from run import report
reviews={
 'CryptBB':dict(scope='First six unselected held-out examples in preview.png.',
    improvements=['Background brightness is closer to the target examples.'],
    failures=['Text remains too thin and stylistically different.', 'Hybrid starting lengths are visibly too short and are frozen by this MVP.', 'Text palettes do not reproduce the targets diverse per-character colors.', 'Coarse grids, blobs and noise differ from target background structure.'],
    conclusion='CryptBB reconstruction remains poor despite improved comparison distances; background brightness improves, but typography, length and clutter remain wrong.'),
 'DarkBay':dict(scope='First 16 unselected held-out examples in comparison.jpg and first six in preview.png.',
    improvements=['Mixed light/dark text and muted background palette are closer to targets.'],
    failures=['Background retains unrelated grids, noise and interference.', 'Background color distribution is too narrow.', 'Font mixture, glyph displacement and geometry still differ from targets.'],
    conclusion='DarkBay shows visible palette improvement, including light/dark text, but target background structure and typography are not closely reconstructed.')}
for family,review in reviews.items():
    out=U.ROOT/f'{family}_run_01';result=U.read(out/'results.json')
    review.update(real_mask_accuracy='Unknown; no target annotations.',policy_changed_after_test=False)
    result['visual_review']=review;result['conclusion']=review['conclusion'];result['decent_family_reconstruction']=False
    result['distance_reduction_percent']={k:100*(1-result['test']['refined'][k]/result['test']['baseline'][k]) for k in ['text','background','appearance','audit']}
    U.dump(out/'visual_review.json',review);U.dump(out/'results.json',result);report(out,result)
    state=U.read(out/'status.json');state['results']=result;U.dump(out/'status.json',state)
