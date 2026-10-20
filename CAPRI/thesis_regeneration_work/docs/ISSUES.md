# Issues and decisions

| ID | Evidence | Impact | Handling |
|---|---|---|---|
| I01 | Imported `gen_gateonly/generator.py`, `generate()` retry branch | Supplied schema can be replaced on retry | Preserve in baseline; any controlled adapter must keep fitted settings fixed |
| I02 | Same function `_best` state; historical `cryptbb_universal_01/*/result.json` | An earlier image/label can escape a later call | Preserve evidence; never approve stale output for datasets |
| I03 | Imported `generate_lmdb.py`, `generate_one()` | Discards internally returned label | Baseline instrument records both; future adapter must verify alignment |
| I04 | Imported `.font_whitelist.json` and saved generators' `get_font()` | Linux paths unavailable on Windows | Match generic local fonts by filename and verify glyph coverage; log hashes and mapping; no vendor edits |
| I05 | Historical portable previews | C1/C2 styles are generic and not fitted to CryptBB | Reproduction PASS must not be called fidelity PASS |
| I06 | Prior independent renderer `cryptbb_no_vlm_20260927_01_preview` | Distorted glyphs can pass geometric checks | Require visual review; no downstream training claim |
| I07 | Earlier `auto_synth` failure notes | OCR junk and component undercounts | Keep alphabet/count unresolved until validated |
| I08 | Original README claimed no real-data tests | Became stale after previous runs | Preserve old README in `docs/archive`; replace current entry point with current evidence |
| I09 | Initial import preparation assumed `experiments/__init__.py` existed | Preparation failed with FileNotFoundError before publishing | Corrected discovery to allow namespace packages; original repo untouched |
| I10 | Swain paper PDF redirected to university login; color-transfer PDF timed out | Full-text research access incomplete | Cite only verified publisher abstract/bibliography; no full-paper replication claim |
| I11 | External dataset download: Python SSL ImportError; sandbox socket denial | Initial HTTPS acquisition unavailable | Used approved PowerShell download; pinned archive size and SHA-256 verified |
| I12 | External spatial negative control | Identical color histograms survive complete spatial destruction | Restrict Method 1 PASS to measurements; do not claim rendering fidelity from histogram similarity |
| I13 | Original `families.py` and `generate_lmdb.py:generate_one`; [fresh run](experiments/IMPORTED_ALL_DATASETS_01.md) | C1/C2 family configuration constrains text, not visual family style; stored dimensions are not applied; Courier/outlined-text absent | Explicitly mark generic fallbacks and separate rendering execution from regeneration; text priors were supplied, not learned |
| I14 | Fresh run City C2 `012.png`, Courier C2 `012.png`, DarkBay C1 `012.png` | Some glyphs look filled-in despite passing the internal degeneracy gate | Preserve raw outputs and qualitative review; no readability or training-readiness claim |

Code paths in I01–I04 are under `../vendor/imported_repo/`; their exact original paths and SHA-256 hashes are in `../vendor/PROVENANCE.json`. Raw historical runs remain outside this new study.

## Decisions

- 2026-09-27: interpret “south/syth folder” as the existing `E:/pranav/Final sem/sourcefree_synth` folder from the preceding instructions.
- Copy relevant imported files instead of destructively moving the original checkout.
- Maintain separate historical reproduction and new measurement evidence.
- Use assistant vision for qualitative evaluation and record its influence on development. The runtime no-VLM claim is narrower than a completely VLM-free research process.
- No later method starts until every dataset clears the preceding method's declared gate.
