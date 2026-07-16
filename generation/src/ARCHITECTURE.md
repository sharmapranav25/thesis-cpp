# `src/` — Architecture Map (build one part at a time)

This is the blueprint for the VLM-guided source-free captcha-synth **generator**. Each
architecture part gets its own module in `src/`. Build/refine them one at a time against
the **interface contract** below. A working end-to-end **prototype** already exists flat in
`generation/` (built 2026-07-06) — port/reference it per part; don't start blind.

- **Deliverable:** unlabeled captcha folder → labeled synth that matches. No recognizer / no
  training / no eval-scoring. Judged by eye + VLM discriminator. Constraints: `../requirements.md`.
- **Design rationale + research + failure catalogue:** `../DESIGN.md`, `../FAILED_SYNTHESIS_ATTEMPTS.md`.
- **The load-bearing finding so far:** *recreating fonts from generic libraries fails* (no
  library font matches the exotic scheme faces). **Real glyph crops composited onto a
  GENERATED background** is what works — that is the chosen glyph strategy (part 4b).

## Data flow

```
[1] real unlabeled captchas ─► [2] DIRECTOR ─► Spec (difficulty + knobs + colours)
                                                 │
                                   ┌─────────────┴─────────────┐  (split)
                    ─ BACKGROUND branch ─           ─ GLYPH branch ─
        [3a] bg extract + ghost gate        [3b] glyph extract + build labeled bank
        [4a] bg GENERATE (php, VLM knobs)    [4b] glyph SOURCE (real-crop bank / recreate)
                                   └─────────────┬─────────────┘  (merge)
                                        [5] COMPOSE ─► synth.png + label (ours)
```

## Conventions (all parts)

- **Runtime:** venv `E:/pranav/model 1/.venv/Scripts/python.exe`; env `HF_HOME=E:\hf_cache`
  (C: is near-full — never cache to C:), `PYTHONIOENCODING=utf-8`. PHP `C:/xampp/php/php.exe`.
- **Run a module standalone:** `python -m src.<module> ...` from `generation/` (each module
  ships a `__main__` CLI so a part can be exercised in isolation). Edit scope = `generation/`.
- **VLM discipline** (see `vlm.py`): never transcribe; rank-not-score (pairwise + order-swap);
  ordinal buckets for difficulty; gates use k-sample consensus + a classical tripwire; colours
  are read from PIXELS, not the VLM.
- **The contract between parts is data, not calls** — each part reads/writes the typed
  artifacts below, so any one part can be swapped without touching the others.

---

## Shared infrastructure

| Module | Role | Prototype | Key interface |
|---|---|---|---|
| `src/vlm.py` | one lazily-loaded **Qwen2.5-VL-3B** + JSON harness + role prims | `../vlm.py` | `load()/unload()`, `ask_json(prompt,images)->dict`, `ask_choice(prompt,images,choices,k)->(win,votes)`, `is_clean(img,k)->bool`, `which_closer(ref,a,b,aspect)->'A'|'B'|'tie'`, `is_match(ref,cand)->bool`, `validate_str/list(...)` |
| `src/glyph_bank.py` | vendored real-crop bank | `../_glyph_bank.py` | `GlyphBank(scheme_dir).render_char(ch)->Image`, `.to_alpha`, `.observed_chars`, `.cluster_to_char` |
| `src/font_atlas.py` | ~175-font DINOv2 retrieval atlas (for the *recreate* variant of 4b) | `../font_atlas.py` | `build_atlas()`, `FontAtlas().query(crops,top_k)->[(ttf,cos)]`, `render_glyph`, `to_black_on_white` |
| `src/configs/` | `font_pool.json`, `font_atlas.npz`, per-run cfg | `../configs/` | — |
| `src/banks/` | built glyph banks per scheme | `../banks/` | GlyphBank format |

---

## The parts

### [1] Input — look at the real captcha
- **Purpose:** load a sample of the scheme's *unlabeled* captchas + scheme config (dir,
  expected length range, canvas). No processing.
- **Module:** fold into `src/pipeline.py` (`_list_images`, `_median_canvas`, `SCHEMES`).
- **Prototype:** `../pipeline.py` (`SCHEMES`, `_list_images`, `_median_canvas`).
- **Out:** `list[PIL.Image]` (reals), `(W,H)` canvas, scheme cfg `{dir, expected, bank}`.
- **Status:** ✅ trivial/done. Input pools: CryptBB=`CryptBBCaptchas(unsolved)` 600,
  DarkBay=`cryptbb_sourcefree/data/darkbay/pool` 799, City=`City (live)` 2000.
  ⚠️ NEVER use `joey_retitling` as CryptBB input (it is the eval answer key).

### [2] Director — difficulty + structured spec
- **Purpose:** VLM reads the reals → coarse forced-choice attributes mapped to concrete,
  discretised generator knobs; difficulty = ordinal bucket fused with a classical prior;
  colours read from pixels. Replaces hand-authored per-scheme configs.
- **Module:** `src/step2_director.py`. **Prototype:** `../director.py`.
- **In:** `list[PIL.Image]`. **Out:** `Spec` dataclass:
  `difficulty:int(1-5), bg:'pale'|'dark', bg_texture, charset_style, serif, weight, case,
  wave:'none'|'mild'|'strong', noise, colored:bool, colors{bg_hex,glyph_hex,noise_hex},
  length_modes:[(len,weight)]` + **`Spec.to_php_cfg(W,H,fonts)->dict`** (php_gen keys).
- **Interface:** `direct(images, use_vlm=True) -> Spec`.
- **Status:** ✅ works. ⚠️ VLM is noisy on categorical fields — **pixels override** bg-lightness
  and `colored`. **TODO:** infer `case='digits'` when crops are all digits (VLM said `mixed`
  for City); detect long streak-lines → set `noise` (VLM missed City's grey streaks).

### [3a] Background extract + ghost gate  *(BACKGROUND branch)*
- **Purpose:** remove the glyphs from a real captcha to expose the background; gate out any
  ghost-glyph residue. In the chosen pipeline the bg is GENERATED (4a), so 3a is a
  reference/fallback source; the **ghost gate** matters whenever real bg is reused.
- **Module:** `src/step3a_bg_extract.py`. **Prototype:** `../pipeline_stages.py::_bg_plate`
  (mask components + `cv2.inpaint`); ghost-gate primitives in `../vlm.py::is_clean` +
  `../compose`? (classical tripwire TBD). Heavier extractor to reuse:
  `auto_synth/src/part_c/extract_bgs_sam.py` (MobileSAM+LaMa, ckpt `auto_synth/configs/mobile_sam.pt`).
- **In:** real captcha. **Out:** background plate image; `is_clean` verdict.
- **Ghost gate contract:** `ghost_gate(bg)->bool` = CV-saliency (diff/upscale/CLAHE) →
  `vlm.is_clean` k-sample consensus → classical CC/high-freq tripwire (auto-reject).
- **Status:** ⚠️ partial — `_bg_plate` inpaint works but sometimes leaves a glyph; the ghost
  gate is designed but **not yet integrated** (only `is_clean` exists). LaMa bg pools already
  exist at `auto_synth/data/<scheme>/bg_lama` for reference.

### [3b] Glyph extract + build labeled bank  *(GLYPH branch)*
- **Purpose:** segment single-glyph crops from reals; cluster + label them into a
  char→crops bank (the real-glyph source for 4b); infer the length distribution.
- **Modules:** `src/step3b_glyph_extract.py` (segmentation + length) and
  `src/step3b_build_bank.py` (bank builder). **Prototype:** `../extract.py`, `../build_bank.py`.
- **In:** captcha / pool of captchas. **Out:** `char_crops(img)->list[PIL]` (glyph-on-white);
  `length_modes(paths)->[(len,w)]`; a GlyphBank-format bank
  (`alphabet/cluster_XXX/*.png` + `alphabet_manifest.json{alphabet_cluster_ids, cluster_to_char}`).
- **Interface:** `char_crops(img, expected)`, `length_modes(paths, expected)`,
  `build(pool_dir, out_dir, allow, k, expected)`.
- **Status:** ✅ works on **pale-bg** schemes (CryptBB, City). ❌ **weak on dark blocky bg**
  (DarkBay): CC segmentation grabs background blocks → dirty crops → block-mess synth.
  **THE key TODO:** SAM-clean glyph masks (MobileSAM prompted by CC boxes) before clustering,
  so dark-bg glyphs come through clean. Banks: reuse `auto_synth/data/{cryptbb_unsolved,city,
  courier_market}` (City/Courier missing digit '6'); DarkBay bank at `../banks/darkbay` is dirty.

### [4a] Background GENERATE (VLM-guided)  *(BACKGROUND branch)*
- **Purpose:** render a clean background (no glyphs, no ghost residue) from the Spec's
  bg/colour/noise knobs, in the same GD primitives real PHP captchas use.
- **Modules:** `src/step4a_bg_generate.py` (driver) + `src/step4a_php_gen_ext.php` (renderer,
  `bg_only` mode). **Prototype:** `../php_gen_ext.php` (has `bg_only`), `../pipeline.py::_render_bg_pool`.
- **In:** `Spec` (via `to_php_cfg` + `bg_only=True`), `(W,H)`, count. **Out:** list of bg PNGs.
- **php cfg keys:** `W,H,fonts,charset_style,bg,min/max_size_frac,max_rot,
  wave{on,ampx,ampy,per},grid,lines,arcs,dots,colors{bg_hex,glyph_hex,noise_hex},bg_only`.
- **Status:** ✅ works — backgrounds match reasonably (CryptBB pale+grid+speckle, DarkBay dark
  blocks, City pale). Ghost-free by construction. **TODO (optional):** best-of-N render →
  `vlm.which_closer` select vs real → one critic edit + verify (currently a single render).

### [4b] Glyph SOURCE  *(GLYPH branch — CHOSEN: real-crop)*
- **Purpose:** provide `render_char(ch)->glyph` for any char, to spell our text.
- **Module:** `src/step4b_font.py`. **Prototype:** real-crop via `../compose.py::bank_for` +
  `../_glyph_bank.py`; recreate variant via `../font_atlas.py`.
- **In:** char + the scheme's bank (3b). **Out:** a glyph image (real crop, exact font).
- **Two variants:**
  - **real-crop (CHOSEN):** `GlyphBank.render_char(ch)` → actual extracted glyph. Exact font,
    label correct by construction. ✅ works when crops are clean; ❌ garbage when 3b crops are
    dirty (DarkBay) → fix is in 3b (SAM). Tradeoff: uses real pixels + charset limited to the bank.
  - **recreate (deprioritised):** `FontAtlas.query(crops)` → closest generic `.ttf`, render with
    it. ❌ **weak** — no library font matches exotic faces (Parchment for CryptBB, math font for
    DarkBay). Kept only as a fallback / for schemes near a generic family.
- **Status:** real-crop ✅ (gated on clean 3b); recreate ❌ weak.

### [5] Compose — merge → synth + label
- **Purpose:** spell OUR chosen text with 4b glyphs, alpha-composited onto a 4a background,
  with realistic layout (overlap + jitter). Emit synth + label.
- **Module:** `src/step5_compose.py`. **Prototype:** `../compose.py`.
- **In:** bank (4b), bg image (4a), text (sampled from bank charset + 3b length modes).
  **Out:** synth `PIL.Image` + `labels.csv (filename,label)`.
- **Interface:** `compose_one(bank, bg, text, rng)->Image`, `to_alpha_smart(glyph)`,
  `sample_text(bank, length, rng)`.
- **Status:** ✅ works. `to_alpha_smart` keys by the crop's border colour (kills white-box
  halos). ⚠️ minor residual halos on streak-contaminated crops; layout has overlap+jitter.

---

## Orchestration & verification (not architecture parts, but needed)

| Module | Role | Prototype | Interface |
|---|---|---|---|
| `src/pipeline.py` | wire 1→5, route by difficulty (real-crop primary / recreate fallback) | `../pipeline.py` | `run_realcrop(scheme,out,n)`, `run(...)` recreate |
| `src/qa.py` | A/B panels + **VLM discriminator** (the success metric) | `../qa.py` | `panel(scheme)`, `discriminate(scheme,trials)->acc` |
| `src/pipeline_stages.py` | render all 7 stages for one captcha (the 1-2-3a-3b-4a-4b-5 view) | `../pipeline_stages.py` | `build(scheme)` |

**Success bar (recognizer-free):** convincing synth for a scheme by eye + VLM discriminator
near 50%. ⚠️ the 3B discriminator is a **weak judge** (scored broken DarkBay at 50% too) — the
eye is the real arbiter until a stronger judge exists.

## Current scoreboard (prototype)

| Scheme | Synth quality | Blocker |
|---|---|---|
| **CryptBB** | ✅ strong | — (pale bg → clean crops) |
| **City** | ⚠️ decent | missing grey streaks (part 2 noise); crop-box artifacts (part 3b) |
| **DarkBay** | ❌ glyphs fail | dark blocky bg → dirty crops (**part 3b needs SAM**) |

## Suggested build order (one part at a time)
1. `vlm.py` (shared — everything needs it).
2. `step3b` **SAM-clean glyph extraction** — the top blocker (unlocks DarkBay + any dark-bg scheme).
3. `step2_director` case/noise fixes (digits detection, streak detection).
4. `step4a` best-of-N + VLM-select refinement (optional quality lift).
5. `step3a` ghost gate integration (needed only if real-bg reuse is added).
6. `step5` layout/halo polish.
