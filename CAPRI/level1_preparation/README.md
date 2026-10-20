# Level 1 — standard supervised recognition baselines

Seven separate model folders: PARSeq, ViTSTR, SATRN, TrOCR, SVTR, SVTRv2 and CRNN.
ABINet is excluded from the training suite due to its academic/non-commercial terms.
Full upstream repositories include unrelated models; their presence is not permission
to use every bundled component under the repository's top-level license.

## What is ready

Model constructors, per-model configurations, CSV image loaders, native losses,
GPU/CPU training launchers, validation exact-string accuracy/CER, checkpoint saving,
and disposable smoke tests. Check each model's `readiness.json` for actual test results.
No labeled training corpus is supplied. No dataset training has been launched.

Defaults: random initialization, uppercase/lowercase/digits, max length 12, AdamW,
20 epochs, learning rate 0.0003 with warmup/cosine decay, effective batch size 32,
seed 42, RGB normalization, no augmentation. These are starting baseline settings,
not tuned hyperparameters or claimed paper reproductions. Models retain their native
losses and architecture-specific input sizes. TrOCR uses a character vocabulary
instead of its released wordpiece vocabulary; no pretrained weights are loaded.

## Start one model

Create independent train/validation CSV files with headers `path,label`. Images may
be absolute paths or relative to the CSV. The example CSVs are format examples only.

```powershell
Set-Location 'E:\pranav\Final sem\Models\level 1\PARSeq'
.\train.ps1 --train-manifest 'D:\data\train.csv' --val-manifest 'D:\data\val.csv'
```

Use the same frozen splits and character settings across models. Train one model at
a time on the 16 GB RTX A4000. Microbatch defaults are conservative starting points;
the two-image smoke tests do not guarantee memory fit at every chosen batch size.
Training rejects blank/unknown/overlength labels, corrupt images and cross-split
duplicate pixels. It does not silently truncate labels. Full runs write resolved
configuration, dataset hashes, metrics, best/last checkpoints, and environment data.
Checkpoint evaluation: `train.ps1 --evaluate PATH --val-manifest PATH`.
There is no automatic checkpoint resume in this baseline launcher.

## Runtime and portability

The local `.venv` uses Python 3.9 from `C:\ProgramData\Anaconda3` and reads existing
GPU packages from `E:\pranav\model 1\.venv\Lib\site-packages` through
`existing_gpu_packages.pth`. Model source/weights are not reused from old experiments.
New compatibility packages are installed only in this level-1 environment. Therefore
this local environment depends on those interpreter/package paths remaining present.
Local `sitecustomize.py` preloads the environment's SSL/SQLite DLLs to support this
Anaconda-based interpreter. The older environment itself was not modified.
For a standalone environment on another machine, create a fresh Python 3.9 environment
and install `requirements.txt`; keep the pinned source folders. This recreation recipe
is separate from the locally tested reused environment. `environment.json` records
versions actually used. Downloaded local compatibility wheels are under `wheels`.

## Sources and licenses

- PARSeq, ViTSTR: `upstream/parseq`, Apache-2.0 code; CRNN submodule carries MIT.
- SVTR, SVTRv2: `upstream/OpenOCR`, Apache-2.0 implementations.
- SATRN: MMOCR Apache-2.0 components, adapted only for standalone imports/registry
  plumbing under `shared/satrn_vendor`. Copyright headers and license are retained;
  component hashes and changes are recorded in its PROVENANCE.json. Original MIT
  TensorFlow SATRN is also retained as a reference under `upstream/SATRN`.
- TrOCR: installed Hugging Face Transformers implementation (Apache-2.0), based on
  Microsoft's TrOCR architecture. See https://github.com/microsoft/unilm/tree/master/trocr
  and https://github.com/huggingface/transformers/tree/v4.57.6/src/transformers/models/trocr.

`PROVENANCE.json` records exact upstream commits. Preserve upstream licenses/notices
when sharing code. Dataset and pretrained-weight permissions require separate review.
