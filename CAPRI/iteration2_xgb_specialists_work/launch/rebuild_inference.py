"""Replay completed boosted inference and previews after an output-decoding fix."""
import argparse
from pathlib import Path
import torch
from fit_xgb import CHECKPOINT, EXPECTED, read, sha, load_models, infer_real, previews, status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    out = parser.parse_args().out.resolve()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert sha(CHECKPOINT) == EXPECTED
    specs = read(out/'bundle/heads.json')['tasks']
    assert len(specs) == 107
    models = load_models(out, specs)
    directory = infer_real(out, specs, models)
    previews(out, directory)
    assert sha(CHECKPOINT) == EXPECTED
    status(out, 'complete', heads=len(specs), source_checkpoint_unchanged=True,
           real_images=3599, previews=300, italic_decoding='probability >= .5')


if __name__ == '__main__':
    main()
