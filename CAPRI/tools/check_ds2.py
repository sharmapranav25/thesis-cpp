"""Check curriculum transitions and fresh renderer seeds before a long run."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from capri_controller.training.ds2 import bootstrap, ROOT, U
bootstrap()
import collections
import hashlib
import numpy as np
from capri_controller.training.ds2_data import BucketSampler
from capri_controller.pipeline import draw_recipe, usable, R


def main():
    labels=np.array(['easy']*100+['medium']*100+['hard']*100)
    sampler=BucketSampler(labels,48)
    expected=[{'easy':300},{'easy':150,'medium':150},{'easy':45,'medium':75,'hard':180}]
    for counts in expected:
        indices=list(sampler)
        assert dict(collections.Counter(labels[indices]))==counts
    assert list(BucketSampler(labels,48))==list(BucketSampler(labels,48))
    entries=U.read(ROOT/'entries.json');R.install()
    rendered=[]
    for position in [0,100,200]:
        entry=entries[position];seed=10000000000+position*10000000
        row=R.safe_render((draw_recipe(entry['recipe'],seed),entry['controls']))
        assert usable(row,U.read(ROOT/'length_decision.json')['default_length']),str(row.get('error'))
        rendered.append(dict(controller=entry['controller'],seed=seed,sha256=hashlib.sha256(row['image'].tobytes()).hexdigest()))
    U.dump(ROOT/'readiness.json',dict(sampler_transitions_verified=True,deterministic_sampling=True,fresh_seed_renders=rendered,
        training_preflight=U.read(ROOT/'preflight/completion.json'),note='Preflight trained only a disposable3-step fixture; no production training or target labels used.'))
    print('Curriculum transitions, seeded sampling, fresh renders and training preflight passed',flush=True)


if __name__=='__main__':main()
