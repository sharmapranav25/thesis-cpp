# Pilot adapter architecture

**OBSERVED:** The isolated [PHP CLI adapter](../../adapters/php_pilot.php) loads the pinned snapshots without modifying them. The [runner](../../feature_bank/pilot.py) calls a fresh process per case and repeat, retains stderr and actual metadata, checks image decoding, dimensions and returned strings, and records hashes. This is an offline rendering adapter; it never validates a live CAPTCHA challenge.

| Adapter | Mechanism and scope | Limitation |
|---|---|---|
| Gregwar | Subclass hooks record before/after text and actual line-call count; save upstream randomness fingerprint | Complete image, warp couples glyphs and lines; pre-warp mask is not final effect truth; scatter disabled by explicit colors |
| Securimage | Subclass captures text-stage images and saves output without headers/session/database | Seed replay passes but enabling noise consumes randomness before text and changes layout; same seed is not a counterfactual pair |
| MobiCMS | Direct API, generic OFL fonts, font-case tuning explicitly disabled | Uses cryptographic random_int; seeds cannot replay. Qualitative only until actual draws can be captured/replayed |
| simple-php-captcha | Original image route with isolated per-case session, explicit plain background and font | Complete image; positions can clip text or shadow. Rotation changes bounding box and random position ranges |
| Existing Python | Unchanged render_once with explicit generic profile, fonts and saved stage images | Line width fixed at one pixel; outline option is a filled glyph border, not hollow lettering; many effects still lack final visibility masks |

**READ:** Source parameters are not claimed as arbitrary ranges. The pilot tests Gregwar line maximum 0 or 3 (width randomly 1–3), Securimage curve count 0 or 3/noise level 0 or 2/perturbation 0 or 0.65, simple shadow offset (3,3) and absolute angle 0 or 8 degrees, and the exact Python settings saved per case. These sampled values are not proof of support for every value between endpoints.

**OBSERVED:** All five deliberately invalid adapter requests were rejected: negative canvas, nonalphanumeric text, unknown mode, nonexistent font, and unknown option. This is adapter validation, not a claim that each upstream API validates every parameter.

**PROPOSED next repairs:** record random draws and glyph geometry in separate instrumented adapters; replay those draws for effect toggles; enforce clipping rejection for every glyph and shadow; add final visible-effect masks; add observer/noninterference tests. Preserve this first pilot as evidence, and use a new run for repairs. Do not promote requested flags to positive visual labels.

No `bank atlas`, model training, classification, or matching interface is represented as complete by these pilots. Completed entry points are `python -m feature_bank prepare`, `python -m feature_bank audit-freeze`, and `python -m feature_bank.pilot`. The old sourcefree_synth CLI is unchanged.
