# Iteration 2: readable static corpus, design and feasibility pilot

2026-10-03. User authorised this work after reviewing the dataset proposal.
First training is one model on a frozen dataset, with ordinary shuffled epochs.
No curriculum, adaptive data, CNN freezing or specialists. This document concerns
dataset design only; no training is launched by the pilot.

## Quotas for the eventual 100,000 training images

Every image trains both branches. Presence negatives are useful labelled examples,
not wasted images. Balancing means accepted measured labels, not requested settings.

| Dimension | Target | Relationship / qualification |
|---|---|---|
| Effects per image | 33,334 with 3; 33,333 with 4; 33,333 with 5 | 399,999 visible positive effect labels total |
| CNN effect heads | 10,810 or 10,811 positives each, initially aim within 5% | 37 heads. Scribbles' two functions count as one head. Exact feasibility with exclusivity is checked before claiming passage. |
| Length | 14,285 or 14,286 per value 2–8 | Reject a rendered string that does not match its ticket |
| Font category | 20,000 each | Same category when mixing fonts; held-out fonts never used in train |
| Content | 25,000 each: digits, upper, lower, both cases | Actual head semantics: uppercase/lowercase allow digits; mixed means both cases, not necessarily digits. Balance character tokens separately; four groups do not imply equal frequency over all 62 characters. |
| Font weight and width | About 9,091 per attainable measured value 0–10 | Target sampling is not proof of measured balance. Weight needs transformed-mask fitting. Unreachable levels are reported rather than relabelled. |
| Text slant / font slant | Shared activation; about 9,091 slanted tickets, remainder zero in initial pilot | These measure the same applied shear; positive levels audited separately. Uniform 0–10 across all images would make almost every image slanted. |
| Rotation, jitter, size variation, curve, overlap, wide spacing, font mixture, multicolour, two-layer | Initial equal focus-ticket allocation, about 9,091 each, with neutral tickets too | Measured prevalence must be audited. Focus tickets are a feasibility probe, not a promise that every sampled feature is visible. |
| Condensed / expanded | Coupled to font-width fitting | Do not impose independent conflicting quotas; measure their distribution jointly with width. |
| Italic | 25%, allocated jointly with category/x-height/font-mixture requirements | No fallback changes italic status. Remaining weight/width feasibility is audited from the actual renders. |
| Stroke class | Report joint coverage within category | Some category/class combinations are absent; do not invent a Cartesian balance. |
| X-height | Equal shares over supported levels, joint with category and italic status | An integer quota solver uses category/level cells with at least 3 eligible fonts, at least 1 of the required italic status (2 for font mixtures). Accepted images must hit the assigned intrinsic level exactly. Level 9 has too few fonts for this policy. |

## Difficulty controls

The original k35 generator adds independent text alterations on top of its 3–5 effects.
The new pilot starts with neutral text geometry, activates one named text focus, and
fits font width and weight. Condensed/expanded may additionally follow width fitting.
All chosen effects except one focus use requested levels 1–4. The focus spans 1–10.
If a geometric text focus is at requested level 6 or more, damaging effects are capped
at requested level 3. These are proposed engineering constraints, not a proven
readability metric. Low requested levels can still be severe in combination.

For the final dataset, level quotas must be decided from the accepted measured
histograms. The initial pilot deliberately makes strong isolated examples and mild
combinations; it does NOT claim uniform positive levels 1–10. Do not silently describe
requested levels as achieved labels, or claim absent levels are covered.

Pilot guards: original renderer guard; every glyph contrast >=70/255; minimum glyph
height >=12 native pixels; maximum clipping <=5%; original post-glyph-effect retained
share >=80%; final binary-mask stroke width >=1.2 native pixels. The retention denominator starts AFTER thin/hollow operations, so it
cannot certify that those operations preserve glyph identity. Human review is required.

The pilot also requires the final labelled positive effect set to equal the planned
set, so accepted effects-per-image counts and planned head quotas stay traceable.
Existing final ownership maps handle covering by opaque effects, but blur and other
RGB-only operations can leave stale visibility labels. This remains an explicit
full-corpus blocker; current checks do not establish perceptual visibility of every effect.

## Pilot and acceptance

`feature_bank/iter2/balanced.py` uses fixed tickets generated before rendering.
SciPy's existing `milp` solves joint category/x-height/italic/font-mixture quota allocation; no new
dependency was installed. Two-row rendering is scheduled only for lengths >=4.
The renderer's row-transition gap calculation was fixed so it does not falsely label
the transition between rows as tight spacing. Whole-canvas fields precede local shapes.
Each retry retains length, category, content and the chosen effect heads. It may change
fonts, colours, level settings and rendering seed. After 60 attempts, a failed ticket
is reported; no successful-looking complete dataset is declared with missing tickets.
Thus even accepted prevalence can be biased by retries; audit the actual label bins.

Pilot cache is compatible with the existing loader, under `iter2_h96/smoke`.
Hashes cover resized trainer inputs. `balance_report.json` contains planned and
accepted effect counts, measured text prevalence and levels, font distributions,
target-vs-measured accuracy, rejected-attempt reasons, missing tickets and duplicates.
Sampler, font-table and text-renderer hashes are recorded. Explicit readiness gates
remain false until visibility validation and human image review are complete.
Individual review PNGs are about 3x native resolution, with full JSON labels alongside.

PowerShell, from `E:\pranav\Final sem\sourcefree_synth`:

```powershell
$py = 'E:\pranav\model 1\.venv\Scripts\python.exe'
$env:PYTHONPATH = '.'
& $py -B -m feature_bank.iter2.balanced --out runs/iter2_readability_pilot_NEW --n 420 --workers 8
```

Before a full run: inspect readable and difficult individual examples with the user;
resolve final-effect visibility; establish reachable level bins and x-height support;
verify measured text/font quotas and character-token coverage; confirm length/category/
content and effect-count quotas, unique hashes, and split disjointness. Adjust the
static sampler from this evidence, freeze its configuration, then generate the full
corpus in new directories. Dataset construction may fill deficits before freezing;
there is no adaptation during training.

100k is training only. Steering, selection, calibration and seen_test are separate
2k sets; generalisation is 2.5k with held-out fonts, as the current handoff specifies.
All use the frozen approved distribution and disjoint deterministic seeds.
Small pilot results cannot prove the full-corpus balance or model learnability.

## Outcome

Development probes are preserved under
`C:\Users\pranav\Documents\ChatGPT\thesis\iteration2_dataset_work`:

| Probe | Accepted | Weight target within one | Width target within one | X-height target within one |
|---|---:|---:|---:|---:|
| pilot_01 | 134/140 | 93.3% | 97.0% | not controlled |
| pilot_02 | 140/140 | 87.9% | 98.6% | 78.6% |
| pilot_03 | 415/420 | 87.5% | 91.6% | 100% |

Pilot_03 used joint category/x-height quotas and the stricter contrast guard.
It exposed missing weight level 0, underrepresented weight level 10, an achieved
italic share of only 14.5%, five unfilled tickets, and false tight-spacing labels
on two-row text. Those failures are preserved, not promoted as balanced training data.
The two-row label bug and a font-mixture candidate-count issue were fixed afterwards.
The final published pilot is measured separately; do not reuse the earlier numbers
as evidence for its changed code.

No full generation or training follows numerical checks alone; the agreed
individual-image review happens first. Final-image perceptual effect visibility,
measured weight-level balance, and italic feasibility remain open. The pilot CLI
currently writes a smoke cache only; it is not yet a full six-split corpus builder.

The subsequent v5 fixed schedule makes italic status part of the integer allocation,
rather than an independent draw with a fallback. The 100k schedule is feasible with
exactly 25,000 italic tickets and 10,000 tickets for each supported x-height level;
font mixtures have at least two matching fonts. This is scheduling evidence only;
accepted-image checks and the final pilot determine actual coverage.

## Published review pilot: v5 results

`runs/iter2_readability_pilot_20261003_02/`, 420 fixed tickets, 418 accepted.
Earlier published v4 pilot remains in `runs/iter2_readability_pilot_20261003_01/`
(416/420); do not overwrite it or treat its numbers as the v5 outcome.

- CNN positives: 43–46 per head (37 heads), each accepted image has exactly its
  assigned 3/4/5 visible-labelled effects. Balanced appearance does not yet certify
  final RGB perceptual visibility after degradation.
- Controlled text appearance heads: 37–39 positives each; neutral focus also included.
  Tight spacing now has 38 positives after the two-row measurement correction.
- Width-linked condensed/expanded: 162 / 186 positives. These remain much more common
  than the independently controlled text features; joint width/appearance redesign is
  required if equal positive prevalence across every transformer head is demanded.
- Length: 59–60 per value; categories: 83–84 each; content groups: 104–105 each.
- Italic: 104/418 (24.9%); x-height hits assigned levels exactly on all accepted images.
- Weight within one of target: 85.2%; width: 91.4%. Actual weight level 0 absent;
  level 10 has 19/418 (4.5%), below the earlier 5%-per-level requirement. Within-one
  accuracy does not establish an even measured level distribution.
- No duplicate cached images. Existing cache loader and an 8-image finite-label batch
  checked. Four unit regression checks pass, including the 100k coupled quota schedule.
- Tickets 20 and 212 failed all 60 attempts. Both combine thin target weights with
  multiple text-damaging effects; they are retained in the failure report, not relabelled
  or substituted silently.

**Full corpus is not ready.** Remaining work: user readability review; measurable
final-effect visibility; reachable weight bins and accepted-level balancing; explain
or redesign width-linked prevalence; and fill/reallocate failed ticket combinations
without losing quota margins. No 100k data or training was started.

Read `balance_report.json` for complete measured histograms and failed tickets.
Review one large PNG at a time under `review/`, with complete labels in its JSON file.
