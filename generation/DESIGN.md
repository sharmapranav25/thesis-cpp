# DESIGN — VLM-Guided Source-Free Captcha Synthesis (v2)

> The plan, not the code. This is the "clear idea of what needs to be done" before
> anything gets built. It restates your matured architecture, grounds each step in
> (a) what the literature says actually works + the traps, and (b) what code already
> exists in the repo to reuse, then lists the genuinely-new pieces, the forks that are
> yours to call, and the build order.
>
> Constraints it must satisfy: [requirements.md](requirements.md) (source-free, label-free,
> generic PHP/font libs OK, local 16GB A4000). What already failed and why:
> [FAILED_SYNTHESIS_ATTEMPTS.md](FAILED_SYNTHESIS_ATTEMPTS.md).

---

## 0. TL;DR

Point at a folder of **unlabeled** captchas → one **VLM (Qwen2.5-VL)** sits in the loop
the whole way and (1) rates difficulty + emits a **structured parameter spec**, then the
pipeline **splits**: **Branch A** rebuilds the *background* with a parametric generator the
VLM nudges (and a VLM+classical **ghost gate**), **Branch B** rebuilds the *font* by
**retrieving the closest generic library font** to the extracted glyphs (VLM tags +
font-embedding match, not diffusion) → the two branches **merge** in a renderer that stamps
**our chosen text** (label correct by construction) → mass-produce → train a recognizer →
**score on real** (the R6 test auto_synth never ran).

**The two big changes from `auto_synth`:**
1. **VLM-in-the-loop *everywhere*** (director → gate → guided generator → critic), not
   "a VLM bolted onto Part C."
2. **Branch B *recreates* a font from generic libraries** instead of re-pasting **real
   glyph pixels**. That fixes auto_synth's two flaws — the R1 smell (reusing real glyphs)
   and the coverage gaps (digit '6' never recovered) — at the cost of having to *match*
   the style, which is what the VLM+embedding matcher is for.

**The honest tension** (§6, decision 1): real-glyph reuse is what made auto_synth *visually*
match; recreating from libraries is cleaner and fully-covered but may match an exotic font
(CryptBB) *less* well. We mitigate with a font-embedding matcher + distortion, and keep
real-crop reuse as a fallback the difficulty router can invoke.

---

## 1. Architecture

```
                          real UNLABELED captcha  ─────────────┐
                                    │                          │ (whole pool, for
                          [1] look  │                          │  attribute voting)
                                    ▼                          │
   [2] VLM DIRECTOR ──────────────────────────────────────────┘
        emits: difficulty tier (ordinal)  +  structured JSON spec:
          bg{type: solid|gradient|noise|texture|lines, colors[], noise_density}
          font{serif|sans|mono, weight, slant, stroke, colored?, per-char color?}
          distort{wave_amp, rotation, overlap}   length_hint
                                    │
                                    ▼
   [3] VLM-guided LETTER EXTRACTION   (DINO-attention / SAM, routed by the VLM's color/style)
                    ┌───────────────┴───────────────┐
             extracted background            extracted glyph crops     (+ char count → length dist)
                    │                                │
        ┌───────────▼───────────┐        ┌───────────▼──────────────┐
        │      BRANCH A          │        │       BRANCH B            │
        │      BACKGROUND        │        │       FONT / GLYPH        │
        ├────────────────────────┤        ├───────────────────────────┤
  [3a]  │ VLM + classical GHOST  │  [3b]  │ font-attr extract:        │
        │ GATE → reject residue, │        │   VLM coarse tags  +      │
        │ retry / re-inpaint     │        │   FontCLIP/DeepFont embed │
  [4a]  │ VLM-GUIDED parametric  │        │ → RETRIEVE closest generic│
        │ bg generator (PHP-GD): │  [4b]  │   font from FONT ATLAS    │
        │ VLM sets knobs,        │        │ render glyph + distort    │
        │ best-of-N, VLM selects │        │ (color/wave = render-time)│
        │                        │        │ → INSPECT match (embed +  │
        │                        │        │   VLM pairwise A/B)        │
        └───────────┬────────────┘        └───────────┬───────────────┘
                    └───────────────┬─────────────────┘
                                    ▼
   [5] MERGE / COMPOSE:  bg  +  glyphs spelling OUR text  →  synth.png + label
                                    │
                                    ▼
        mass-produce (10k+)  →  train PARSeq  →  eval on REAL labeled set  (R6)
```

### Per-step spec

Each step below: **what** · **VLM role** · **research** (what works + the trap, with a
citation) · **reuse** (existing code) · **new** (what we build).

---

**[1] Look at the real captcha** — sample a handful (and the whole pool for voting).
Nothing to build; it's the input to [2].

---

**[2] VLM DIRECTOR — difficulty + structured param spec** *(new — the biggest missing piece)*
- **What:** the VLM reads the real captcha(s) and emits (a) an **ordinal difficulty tier**
  to allocate effort, and (b) a **structured JSON spec** that seeds every downstream
  generator knob. This kills the hand-authored per-scheme `SCHEME_CFG`/`PROFILES` tables —
  the thing that makes today's code *not* point-at-a-folder.
- **VLM role:** structured-JSON emitter (LMD/LayoutGPT pattern — the LLM emits a *spec*, not
  a prompt) + ordinal difficulty rater.
- **Research:**
  - *Emit a spec, not free text* — **LMD / LayoutGPT** (Lian, TMLR'24; Feng, NeurIPS'23).
    Our generator is deterministic PHP-GD, so a JSON spec maps cleanly to CLI args.
  - *Difficulty as ordinal buckets, not a number* — **Q-Align** (Wu, ICML'24): ask for
    `very-easy…very-hard`, read the **logprobs over the level tokens** for a smooth score.
    Fuse with a **free classical complexity prior** (edge/stroke density, distinct-color
    count, bg variance, warp estimate; Saraee CVIU'20) and default to *more* effort on
    disagreement.
  - *Trap:* general-VLM "difficulty" tracks visual busy-ness, not *recreation* difficulty —
    anchor the prompt to concrete recreatable factors (warp, color count, texture).
- **Reuse:** `captcha regenerator/masking/src/masking/stage0.py::characterise` — already
  returns validated JSON (`char_colors`, `noise_color`, `noise_types`, `char_style`) with
  the repo's best JSON harness (fence-strip + regex + allow-list + fallback + SHA cache).
  **Extend** its schema with difficulty + bg/distort fields.
- **New:** the difficulty prompt + the classical prior + the spec schema.

---

**[3] VLM-guided letter extraction (DINO-attention / SAM)**
- **What:** remove+extract the characters → yields the **background plate** (Branch A) and
  the **glyph crops** (Branch B) and a **char-count** (length distribution for [5]).
- **VLM role:** routes the extractor (the VLM's color/style tells us HSV-path vs
  attention-path; its text description prompts GroundingDINO).
- **Research:** pretrained scene-text *removers* (PSSTRNet) fail on captchas (all-background
  — see failure catalogue); DINO-attention + SAM + LaMa is the workable route.
- **Reuse (this is already built and VLM-routed):**
  - `masking/stage2.py` — **DINO ViT-S/8 self-attention** + Canny masking (Path B) / HSV
    (Path A), routed by stage0's color → **REUSE-AS-IS** (this *is* your step 3).
  - `masking/stage2_5.py` — coverage validator (self-correcting) → **REUSE-AS-IS**.
  - `auto_synth/src/part_c/extract_bgs_sam.py` — **MobileSAM box-prompt + LaMa** (the only
    SAM wrapper in the repo, ckpt at `auto_synth/configs/mobile_sam.pt`) → **REUSE-AS-IS**.
  - `auto_synth/src/part_c/extract_bgs_grounded.py` — iterative GroundingDINO+SAM+LaMa with
    VLM prompt/oracle → **MODIFY** (convert its oracle from loop-terminator to the [3a] gate).
  - `cryptbb_sourcefree/src/background.py` char-removal for reference.
- **New:** nothing structural; just wire the VLM routing signal from [2] into the extractor.

---

**BRANCH A — BACKGROUND**

**[3a] VLM ghost gate — reject character residue** *(new gate; primitive exists)*
- **What:** every extracted background is screened for **leftover glyph/underline residue**
  (the Courier-underline / ghost-letter failure). Reject → retry extraction / re-inpaint.
- **VLM role:** binary accept/reject — *second opinion*, not sole detector.
- **Research (this is the highest-risk check):**
  - **"Seeing Isn't Believing"** (arXiv:2604.21523): evaluator VLMs miss subtle residue
    **>50%** of the time and are *conservative* (biased to say "clean"). ⇒ **do not lead with
    raw VLM perception.**
  - Recipe: **(1)** make the artifact salient with cheap CV first (diff-image vs local
    inpaint, upscale 2–4×, CLAHE), **(2)** ask a *decomposed binary* ("any partial
    letter/stroke edge? yes/no"), **(3)** **Consensus / self-consistency** — k=5 samples at
    T≈0.7, accept only if unanimous-clean (Consensus Entropy, arXiv:2504.11101), **(4)** a
    **classical tripwire** (residual connected-component / high-freq energy in the masked band)
    that auto-rejects regardless of the VLM.
- **Reuse:** `cryptbb_sourcefree/src/background.py::LamaBGPool` + `_has_ghost_glyph` — keep
  the load/filter/sample **harness**, **replace** the blob heuristic with the VLM call.
  `auto_synth/src/part_c/vlm_qwen.py::is_clean` — the VLM oracle **already exists**, lift it
  as the gate.
- **New:** the saliency-boost CV + consensus wrapper + classical tripwire around `is_clean`.

**[4a] VLM-guided parametric background generator** *(new guidance; generators exist)*
- **What:** instead of re-pasting a real background crop (auto_synth), a **parametric**
  generator produces the bg, with the **VLM setting the knobs** ("plain solid? which colors?
  texture? density?") and nudging via best-of-N.
- **VLM role:** sets baseline params [from step 2] → **selects best-of-N** renders (VQA
  selector) → emits a **directional param edit**, then a **verify** pass.
- **Research:**
  - **DreamSync** (Sun, NAACL'25) — label-free: render N candidates, VLM **selects** the best;
    gradient-free, no annotations (matches R3).
  - **Idea2Img** (Yang, ECCV'24) / **SLD** (Wu, CVPR'24) / **LL3M** (2025, VLM critic over
    *parametric* renders) — the critic→edit→re-render→**verify** loop for a *non-differentiable*
    generator.
  - *Traps:* free-form float edits **oscillate** → **discretize knobs to a grid** and prefer
    **best-of-N selection** over free edits; VLMs **hallucinate exact colors** → sample colors
    by reading pixels back from the real image, not from the VLM; **do NOT** use RL/diffusion-LoRA
    (DDPO/DPOK) — wrong tool for PHP-GD, wastes the 16GB.
- **Reuse:** `cryptbb_sourcefree/src/php_gen.php` (+`php_pool.py`) — most complete parametric
  GD renderer (imagettftext + wave + noise over pale/dark bg), **MODIFY** so the VLM emits its
  config instead of hand-authored `SCHEME_CFG`. Also `background.py::procedural_bg/dark_block_bg`
  and `distort.py::PROFILES` — **MODIFY** constants → VLM-set knobs. `background.py::composite`
  → **REUSE-AS-IS**.
- **New:** the best-of-N + select + critic-edit + verify controller.

---

**BRANCH B — FONT / GLYPH** *(the biggest new capability; auto_synth never recreated a font)*

**[3b] Font-characteristic extraction → retrieve closest generic font**
- **What:** with no labels, extract the *style* (serif/sans, weight, slant, stroke,
  colored?, wavy?) and **pick the closest font from a generic library atlas** — a real,
  reusable `.ttf`, so we can cheaply render thousands of correctly-labeled strings.
- **VLM role:** **coarse forced-choice tags only** + whole-pool majority vote. The
  *quantitative* match is done by a font-specialized embedder, not the VLM.
- **Research (this shapes the whole branch):**
  - **"Texture or Semantics? VLMs Get Lost in Font Recognition"** (arXiv:2503.23768): best
    VLM ~**31%** font-ID, most open models <10%, and they attend to *text* not glyph shape.
    ⇒ **never ask the VLM "what font is this" or for an absolute similarity score.**
  - **FontCLIP** (Eurographics'24) — CLIP tuned for typography; query fonts by attribute
    words *and* by a glyph image in one shared space. **DeepFont** (ACM MM'15) / **Paired-Glyph
    Matching** (BMVC'22) — font-similarity embeddings that **beat CLIP/DINOv2** on font
    retrieval. These do the quantitative matching.
  - *Color & waviness are **render-time distortions, not font attributes*** — folding them
    into font choice corrupts retrieval; handle them in PHP-GD ([4b] distort).
  - *Font embedders are trained on clean text* → de-warp/binarize/normalize the captcha
    glyph crops before embedding, or retrieval collapses.
- **Reuse:** `stage0.py::char_style/char_colors` (already extracted, currently *dropped* by
  `api._pick_color` — wire it in). `model 0/generators/lib_common.php::font_pool()` — **~24
  vendored lib TTFs + ~100 Windows fonts** = the atlas source (**REUSE-AS-IS** as the arsenal).
  `auto_synth/src/part_a/embed.py::DinoEmbedder` — embedder scaffold (**MODIFY** → FontCLIP/DeepFont).
- **New:** build a **font atlas + queryable index** (render each library font's glyph sheet,
  embed, NN-index, tag serif/weight/slant); the VLM tag → retrieval fusion.

**[4b] Render glyph in the chosen font + distort, then VLM/embedding inspects the match**
- **What:** render OUR characters in the retrieved font, apply the render-time distortions
  (color, sine-wave, rotation, overlap), then **confirm the recreated glyph matches** the real.
- **VLM role:** **pairwise A/B only** ("which is closer to the reference?"), order-swapped;
  embedding cosine does the heavy filtering, VLM breaks near-ties.
- **Research:** **"MLLM-as-a-Judge"** (Chen, ICML'24) + **"VLM Judges Can Rank but Cannot
  Score"** — VLMs are trustworthy at **pairwise ranking**, unreliable at absolute scores;
  always **swap A/B order and require agreement** (position bias). Fallback generator if no
  atlas font clears the bar: **FontDiffuser / DG-Font** (per-glyph *images*, expensive, glyph-
  coverage risk) — last resort only.
- **Reuse:** `cryptbb_sourcefree/src/render_source.py::SourceRenderer` (glyph layer + mask +
  label; **MODIFY** font choice from `random.choice` → retrieved font). `distort.py` /
  `php_gen.php` wave+noise. `auto_synth/src/part_b/glyph_bank.py` — keep its
  `render_char()`/`to_alpha()` **API shape** so [5] is unchanged, **replace** its real-crop
  internals.
- **New:** the embedding-cosine match scorer + VLM pairwise tie-breaker (the mirror of `is_clean`,
  call it `is_match`).

---

**[5] MERGE / compose → synth + label**
- **What:** the two validated branches meet: sample a length (from the real distribution),
  choose **our text**, render its glyphs (Branch B) onto the background (Branch A), emit
  `synth.png` + label. **Label correct by construction** (we chose the text) — this is what
  satisfies R3.
- **Reuse:** `auto_synth/src/compose.py` (**MODIFY** — keep skeleton + length-mode sampling,
  swap the two data sources to the new branches; fix the *naive even-slot layout* → add
  wave/overlap to match real geometry). `infer_length.py` / `infer_bank.py` /
  `expand_bank()` → **REUSE-AS-IS** for length + charset. `background.py::composite` merge op.
- **New:** better layout (wave/overlap); wire branch outputs in.

---

## 2. Cross-cutting VLM discipline (non-negotiable, straight from the judge research)

These apply to **every** VLM call and are the difference between this working and looping on
phantom judgments:

1. **Never ask the VLM to transcribe** the captcha. It's unreliable (≤2.5% word) *and* using
   read text as a label violates R3. VLM = style/attributes + accept/reject/rank only.
2. **Rank, don't score.** Pairwise A/B with **order-swap + agreement**, never "rate 0–100."
3. **Difficulty = ordinal buckets + logprob expectation**, fused with a classical prior.
4. **Behind every subtle-artifact gate, a classical tripwire** (CC/high-freq energy). The VLM
   is a second opinion; a blind VLM must not silently pass ghost text.
5. **Make artifacts salient before asking** (diff-image, upscale, contrast) — we know the mask.
6. **Consensus / self-consistency** (k≈5 @ T≈0.7) turns one fallible opinion into
   accept/reject/**abstain**; abstain escalates (re-extract / re-gen), never forces a call.
7. **Read colors from pixels**, not from the VLM's memory (it hallucinates hex).
8. **Discretize generator knobs to a grid**; prefer best-of-N select over free-float edits
   (the #1 cause of critic-loop oscillation). Cap loops at 3–5 iters + a verify pass.
9. **The VLM never sees the eval set** (R7/R8) — only the unlabeled pool.

---

## 3. What we reuse vs build (summary)

**~90% of the mechanical parts already exist.** The gap is the VLM *control loop*.

| Matured component | Source file | Verdict |
|---|---|---|
| VLM structured-JSON harness | `masking/…/stage0.py` | MODIFY (add difficulty + bg/distort fields) |
| DINO-attention extractor | `masking/…/stage2.py`, `stage2_5.py` | REUSE-AS-IS |
| SAM+LaMa extractor | `auto_synth/…/extract_bgs_sam.py`, `grounded_dino.py` | REUSE-AS-IS / MODIFY |
| Ghost oracle (`is_clean`) | `auto_synth/…/vlm_qwen.py` | REUSE (wrap w/ CV+consensus+tripwire) |
| Ghost-gate harness | `cryptbb_sourcefree/…/background.py::LamaBGPool` | MODIFY (swap blob test → VLM) |
| Parametric bg + font renderer | `cryptbb_sourcefree/…/php_gen.php`, `distort.py`, `render_source.py` | MODIFY (VLM sets knobs) |
| Font arsenal (24 lib + ~100 sys TTFs) | `model 0/generators/lib_common.php` | REUSE-AS-IS (as atlas source) |
| Glyph embedder | `auto_synth/…/part_a/embed.py` | MODIFY (→ FontCLIP/DeepFont) |
| Compose / merge + label | `auto_synth/…/compose.py`, `infer_length.py`, `infer_bank.py` | MODIFY / REUSE |

**Genuinely NEW (5 pieces):**
1. **VLM Director** — difficulty + spec (step 2).
2. **VLM ghost gate** — CV-saliency + consensus + tripwire around `is_clean` (3a).
3. **VLM-guided bg loop** — best-of-N select + critic-edit + verify (4a).
4. **Font atlas + retrieval + `is_match` critic** — the whole recreate-a-font branch (3b/4b).
5. **One shared `vlm.py`** — a single lazily-loaded Qwen2.5-VL instance exposing 4 thin
   role-prompts (today the loader is duplicated across 4 files).

---

## 4. Requirements check (does the design obey [requirements.md](requirements.md)?)

| Rule | How the design complies |
|---|---|
| **R1 source-free** | Fonts come from the **generic atlas** (never the scheme's font); bg/glyph *style* is *inferred from unlabeled output images*, which R1 permits. Retrieving "closest generic font" ≠ using the real font. |
| **R2 generic libs OK** | Generator = PHP-GD + the 5 generic PHP captcha libs + system fonts. ✅ |
| **R3 no labels received** | Labels are only **our rendered text**. VLM is forbidden from transcribing (§2.1). No human/eval labels enter. ✅ |
| **R4 generality** | The VLM Director replaces the hand-authored per-scheme configs → truly point-at-a-folder. ✅ |
| **R5 local/16GB** | One Qwen2.5-VL-3B (fp16 ~7.5GB) + SAM/LaMa/FontCLIP co-resident on the A4000; no cloud, no RL. ✅ |
| **R6 validation** | The pipeline **ends in the gold-standard test** auto_synth skipped (§5). |
| **R7/R8 data** | VLM + extractor see only the unlabeled pool; eval sets scored-only; CryptBB input = `cryptbb_unsolved` (disjoint), never `joey_retitling`. ✅ |

---

## 5. Success test (the thing auto_synth never did)

Per your call — **it must make working synth for all three of CryptBB, DarkBay, City**, or
it's not working. For each scheme:

1. Run the pipeline on the **unlabeled** input pool → produce ≥10k labeled synth.
2. Train a PARSeq (reuse `model 1`/`model 2` trainer) on that synth.
3. Eval on the **held-out labeled real** set; report **word + char** acc.

| Scheme | Unlabeled input | Eval set | Bar to clear |
|---|---|---|---|
| CryptBB | `cryptbb_unsolved` (600, disjoint) | 651 (`model 1/data/eval`) | > 30.4% word (source-free VLM); stretch → 57.60% flagship |
| DarkBay | needs a pool (see decision 4) | `DarkBay_Labeled` split | source-free VLM hit 91% — strong bar |
| City | `auto_synth/data/city` input | City labeled split (filename=label, needs confirming) | high (City is the easiest / best visual match) |

**Verdict rule:** "working" = beats the ≤2.5% OCR floor decisively on all three, and is in
the ballpark of (or beats) the existing source-free numbers on CryptBB/DarkBay.

---

## 6. Forks that are yours to call (before any code)

> **DECIDED (2026-07-06):**
> **(1)** Branch B = **recreate-from-libraries primary + real-crop-reuse fallback** (the
> difficulty router invokes the fallback only when no atlas font is close enough on an
> exotic scheme like CryptBB).
> **(2)** VLM = **Qwen2.5-VL-3B**. Already on disk at `E:\hf_cache` (relocated from C: —
> no download needed; verified it loads offline from E:). All caches pinned to
> `HF_HOME=E:\hf_cache` (C: is nearly full — never cache to C:).
> **(3)** Generator = **PHP-GD for the final mass pool + Python `distort.py` for in-loop
> previews** (no PHP round-trip per VLM iteration).
> **(4)** DarkBay pool = *defaulting to* a disjoint split of `DarkBay_Labeled` (≈700 input /
> 300 eval) unless you override — the only fork not yet explicitly confirmed.

**Decision 1 — Branch B: recreate-from-libraries vs real-crop reuse vs hybrid.**
Your architecture says *recreate the font from libraries* (R1-clean, full charset coverage,
but must match style). auto_synth *reused real crops* (best visual match, but R1-smell +
missing '6'). **My rec:** recreate-from-libraries as primary (it's the honest source-free
story and what makes this novel), with real-crop reuse as a **fallback the difficulty router
invokes for exotic fonts** (CryptBB) where no atlas font is close enough. This is the single
biggest design bet.

**Decision 2 — VLM model.** Qwen2.5-VL-**3B** is the sweet spot (fits fp16 with headroom for
SAM+LaMa; the repo already targets it in `extract_bgs_vlm.py` / `model 1/infer.py`) but is
**not on disk** — needs a one-time ~7.5GB download to `E:/hf_cache`. **2B** is local + proven
but a weaker describer; **7B** won't fit fp16 alongside SAM/LaMa (no bitsandbytes/awq
installed). **My rec:** download 3B, standardize everything on it.

**Decision 3 — generator substrate.** **PHP-GD** (`php_gen.php`, most faithful + batchable,
R2-native) vs **Python** (`background.py`+`render_source.py`+`distort.py`, more integrated).
**My rec:** PHP-GD for the final renderer (fidelity + speed), Python `distort.py` as the
in-loop preview during the VLM nudge (no PHP round-trip per iteration).

**Decision 4 — DarkBay input pool.** auto_synth has no DarkBay. Options: reuse
`cryptbb_sourcefree/data/darkbay` unlabeled pool, or split `DarkBay_Labeled` (1000) into
unlabeled-input + held-out-eval (ignoring labels on the input half). **My rec:** split
`DarkBay_Labeled` disjointly (e.g. 700 input / 300 eval) — cleanest and self-contained.

---

## 7. Build order (when we DO start — not yet)

Sequenced so each stage is testable and the highest-leverage / highest-risk pieces come first:

1. **`vlm.py`** — one shared Qwen2.5-VL-3B + the 4 role-prompts + stage0's JSON harness. *(unblocks everything)*
2. **VLM Director (step 2)** — difficulty + spec; kills hand-authored configs. *(makes it point-at-a-folder)*
3. **Branch B: font atlas + retrieval + `is_match` (3b/4b)** — the biggest new capability & biggest R1 win; **highest risk** (exotic-font match), so prove it early on CryptBB.
4. **Branch A: ghost gate (3a)** — cheap drop-in; fixes the known residue failure.
5. **Branch A: VLM-guided bg loop (4a)** — best-of-N + critic + verify.
6. **Merge (step 5)** — wire branches into `compose`, fix layout.
7. **R6 validation** — train + score on CryptBB / DarkBay / City. The go/no-go.

Each numbered item = one reviewable increment. We agree on §6 first, then start at (1).

---

## 8. References (all verified via web this session)

- **Idea2Img** — Yang et al., ECCV 2024, arXiv:2310.08541
- **SLD (Self-correcting LLM-controlled Diffusion)** — Wu/Lian et al., CVPR 2024, arXiv:2311.16090
- **LMD / LayoutGPT** — Lian et al., TMLR 2024 (2305.13655); Feng et al., NeurIPS 2023 (2305.15393)
- **DreamSync** — Sun et al., NAACL 2025, arXiv:2311.17946
- **LL3M** — Sun et al., 2025, arXiv:2508.08228
- **FontCLIP** — Tatsukawa et al., Eurographics 2024
- **DeepFont** — Wang et al., ACM MM 2015, arXiv:1507.03196; **Paired-Glyph Matching** — Cho et al., BMVC 2022
- **Attr2Font** — Wang et al., SIGGRAPH 2020; **FontDiffuser** — Yang et al., AAAI 2024; **DG-Font** — CVPR 2021
- **"Texture or Semantics? VLMs Get Lost in Font Recognition"** — arXiv:2503.23768, 2025
- **Consensus Entropy** — Zhang et al., arXiv:2504.11101, 2025
- **MLLM-as-a-Judge** — Chen et al., ICML 2024; **VLM Judges Can Rank but Cannot Score** — arXiv:2604.25235
- **Seeing Isn't Believing (evaluator-VLM blind spots)** — arXiv:2604.21523, 2026
- **Q-Align** — Wu et al., ICML 2024, arXiv:2312.17090
- **Pairwise or Pointwise? (judge bias)** — arXiv:2504.14716, 2025
```
