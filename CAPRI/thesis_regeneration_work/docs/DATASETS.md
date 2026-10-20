# Dataset register

Location configuration: [datasets.json](../configs/datasets.json). Counts below were read from image files on 2026-09-27, before decoded-pixel deduplication. They are not necessarily unique images.

| Family | Input image files | External held-out pool |
|---|---:|---|
| CryptBB | 600 | `E:/pranav/model 1/data/eval` (labels not read) |
| City | 2,000 | None declared |
| Courier Market | 2,510 | None declared |
| DarkBay | 799 | `E:/pranav/cryptbb_sourcefree/data/darkbay/eval` (200 images) |

Total input image files: 5,909. DarkBay originated in a labeled collection, but label information must not enter the measurement pipeline. JoeyRetitling is not treated as an additional family; existing corpus configuration identifies it as CryptBB evaluation material. MGM is named in the imported family registry but a local image pool has not been verified.

Evidence for configured roles: `E:/pranav/cryptbb_sourcefree/configs/corpus.yaml`; note that its old blocked DarkBay comment conflicts with its active split paths. Use explicit pool/evaluation paths and observed overlap checks, not that stale comment.

Each executed method records input hashes, decode outcomes, duplicate counts, used counts, split membership, per-image results and exclusions. Never use filenames as semantic features. Read all images unless a run explicitly declares a sample limit; qualitative contact sheets are samples even when quantitative analysis covers the entire pool.

## Additional public-data probe

[muddokon/outlined-text-captcha](https://huggingface.co/datasets/muddokon/outlined-text-captcha), pinned revision `6c6a6a6911ae5b3a288378497cb9139e430dee7b`: downloaded its complete upstream training ZIP, 800 images, declared MIT license. Stored under `data/external/outlined_text_captcha/` with labels/filenames excluded from analysis. All 800 are unique by decoded-pixel identity and have no exact overlap with the six checked existing input/evaluation directories. This is an exploratory convenience selection, not a statistically random sample of CAPTCHA datasets or a confirmed real-world source. See [the external study](experiments/EXTERNAL_PIXELS_OUTLINED_01.md). The four primary pools and their sequential gates remain unchanged.
