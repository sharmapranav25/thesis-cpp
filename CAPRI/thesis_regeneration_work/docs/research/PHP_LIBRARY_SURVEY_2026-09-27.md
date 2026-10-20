# PHP CAPTCHA resource survey

Date: 2026-09-27. Status: documentation/source inspection and visual review of published examples; proposed selection protocol, not a completed renderer benchmark.

## Scope and evidence labels

The purpose is to identify reusable rendering capabilities for the proposed feature-classification and recipe-selection pipeline. This survey does not identify the generators behind any real dataset.

- **READ**: a statement supported by linked project documentation or inspected source.
- **OBSERVED**: an appearance visible in a publisher's sample image, inspected in the browser.
- **PROPOSED / INFERRED**: our experimental design or interpretation, not a demonstrated result.

Eight candidates are described below. This is a discovery shortlist, not an exhaustive count of PHP CAPTCHA packages or independent implementations. A known fork and framework components are included. For example, [GregwarCaptchaBundle](https://github.com/Gregwar/CaptchaBundle) explicitly wraps Gregwar/Captcha, so it should not count as another renderer. Cloud verification clients and CAPTCHA-solving service clients are outside this resource pool.

No candidate was installed, executed, or trained on for this survey. Existing project code and dataset files were not changed. Linked branch URLs are mutable: exact revisions, dependencies and asset hashes must be frozen when a candidate is imported. Older published previews demonstrate examples, not current-version reproducibility or complete capability coverage.

## Candidate inventory

| Candidate and primary evidence | READ: relevant capabilities | Proposed role / qualification |
|---|---|---|
| [Gregwar/Captcha](https://github.com/Gregwar/Captcha) | Custom text/font/dimensions, distortion, front/back lines, background images, scatter effect. | First pilot for line and distortion controls. Its API warns that setting a background color disables effects; controls cannot be assumed independent. |
| [Securimage source](https://github.com/dapphp/securimage/blob/master/securimage.php) | `perturbation`, `num_lines`, `noise_level`, colors, fonts, random baselines and optional character boxes. `drawLines()` draws curved interference; `drawNoise()` draws small filled arc marks. | First pilot for noise and deformation. Sample actual draws: enabling random boxes does not guarantee boxes appear. Runtime compatibility remains untested here. |
| [MobiCMS/captcha](https://github.com/mobicms/captcha) | Custom font folders, per-character font mixing, font sizing and image dimensions. | First pilot for typography diversity. OBSERVED: examples contain boxed, circled, dotted and outline-like letterforms. Do not infer an independent shape-overlay API from those appearances. |
| [simple-php-captcha source](https://github.com/yasirmturk/simple-php-captcha/blob/master/simple-php-captcha.php) | PNG background selection, font selection, text size/color, whole-string angle and offset shadow. | First pilot for textured backgrounds and shadows. Asset licensing and modern-runtime compatibility need verification. Distinct project from S1SYPHOS below. |
| [S1SYPHOS/php-simple-captcha](https://github.com/S1SYPHOS/php-simple-captcha) | Explicit Gregwar fork; configurable front/back lines, letter-based noise, scatter, distortion and per-symbol font randomization. | Compare against Gregwar during adapter audit; initially choose one. Shared ancestry makes this a poor independent held-out test for Gregwar. |
| [mews/captcha](https://github.com/mewebstudio/captcha), [configuration](https://github.com/mewebstudio/captcha/blob/master/config/captcha.php) | Laravel integration, font colors, image backgrounds, lines and postprocessing settings. | Reserve candidate. Assess whether its appearance adds coverage worth the framework integration. OBSERVED: preview includes colored text, lines, light/dark backgrounds and an arithmetic challenge. |
| [Laminas Captcha Image](https://docs.laminas.dev/laminas-captcha/adapters/#laminascaptchaimage) | Font/dimension controls, skewing, separate dot and line noise settings. Noise is applied before and after distortion. | Candidate for a renderer withheld from training, after checking ancestry/dependencies. Session/filesystem behavior needs isolation for batch rendering. |
| [Yii 2 CaptchaAction](https://www.yiiframework.com/doc/api/2.0/yii-captcha-captchaaction) | TTF font, foreground/background, character spacing, dimensions, GD/Imagick rendering. GD source varies character size and rotation. | Simple rendering comparison or held-out candidate. INFERRED from the inspected GD path: relatively uncluttered text is a useful negative case for noise classification. |

KCAPTCHA was also located ([Composer adaptation](https://github.com/ganlvtech/kcaptcha)), but its rendering implementation was not assessed sufficiently for the eight-row shortlist. It remains a discovery lead, not a validated resource.

## Published examples reviewed

These are project-published images, not images generated in this session and not our real target datasets. Remote assets were viewed in place, not downloaded into the training pool.

### S1SYPHOS/php-simple-captcha

OBSERVED: varied colors, font shapes, text positioning and crossing strokes; both light and dark backgrounds. It is a Gregwar-derived implementation, not a sample attributed to unmodified Gregwar.

![Publisher's example grid](https://github.com/S1SYPHOS/php-simple-captcha/raw/main/examples.png)

### MobiCMS

OBSERVED: decorated letterforms and variation within a string. Whether a specific visible circle, box or dot pattern comes from a font or a drawing operation must be audited before creating labels.

![Publisher's typography examples](https://github.com/mobicms/captcha/raw/main/docs/images/captcha_example.png)

### mews/captcha

The following asset was read from the Preview image in the project's README and visually inspected. Historical configurations in this image are not guaranteed to match the current configuration file.

![Publisher's Laravel CAPTCHA preview](https://camo.githubusercontent.com/5796b984677c086025788097a4f9f89701595234cb98a23c7276c4ea5424c6ef/68747470733a2f2f696d6167652e6962622e636f2f6b5a784d4c6d2f696d6167652e706e67)

## Proposed selection protocol

The unit of value is an independently controllable rendering capability or a useful different implementation of a capability. Package count alone is not the objective.

1. **Map capabilities.** List text shape/style, rotation, warp, spacing, background texture, lines, dots, decorative shapes and postprocessing. For each operation record its function, parameters, ranges, ordering and coupled side effects. Mark unknowns instead of treating an absent README entry as an absent feature.
2. **Check experimental usability.** Verify offline rendering, valid output text labels, recoverable randomness, generic font provenance and asset licenses. Prefer small adapters that can report actual settings and intermediate layers. Masks and repeatable seeds must be implemented/verified; none are promised by this survey.
3. **Run a small controlled atlas before training.** Use common strings, generic fonts where compatible, and native canvas sizes to compare renderers. Also show native-font examples separately. Produce clean, one-effect, and mixed-effect cases at several strengths. Inspect clipping, contrast, missing glyphs and silent option changes. This is a proposed experiment, not performed here.
4. **Choose by added coverage and cost.** Start with Gregwar (or its S1SYPHOS fork), Securimage, MobiCMS and simple-php-captcha. Keep an addition when it supplies a missing feature, a meaningfully different version of an existing feature, or substantially better controllability. Four is a manageable pilot size, not a scientifically established optimum.
5. **Test generalization.** Withhold an implementation and fonts from training, accounting for forks/shared code. Render the same feature through different sources so a classifier cannot succeed just by recognizing a library. Evaluate on real images too; high synthetic accuracy alone is insufficient.

Feature labels must distinguish appearance from cause. For example: added circular noise, a circle built into a font, and the character O are different cases. Record operations actually drawn and their visibility after composition, not merely requested flags. Parameter estimation (line width, density, location, curvature, etc.) follows feature presence and is needed for useful recreation.

Keep target source code and target font files out of the fitting process. Generic public rendering resources do not imply knowledge of which resource generated a target image. Any alphabet/length inferred from unlabeled images remains a separate unresolved problem.

## Limitations and next evidence needed

- No global package census, library ranking benchmark, runtime compatibility check, training, or regeneration result was produced here.
- Only three published sample images were visually inspected. Other appearance descriptions come from documentation/source and are labeled accordingly.
- The candidate list does not establish support for every feature present in the five existing datasets.
- Pin code and audit each font/background asset before importing resources. Repository-level licensing alone does not settle asset provenance.
- The next concrete experiment would be the controlled atlas and capability checks, followed by a first feature detector only after its labels are reliable.

This entry cites software documentation/source, not research-paper findings. It does not change the completion status of Methods 2-6 or resolve the previously demonstrated target-style regeneration gap. See [the existing imported-generator test](../experiments/IMPORTED_ALL_DATASETS_01.md).
