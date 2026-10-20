# MVP v5 — Controller 2: DINO attention

## Completed outcome

79 profiles retained: City16, CryptBB16, DarkBay16, Outlined15, SimpleCaptcha16. All79 have distinct rendered signatures on a fixed three-recipe probe. No padding or post-evaluation replacement was performed. DINO's diagnostic distance improves22–39% versus hybrid, but visual matching remains partial for City/CryptBB/DarkBay and fails for Outlined/SimpleCaptcha. Synthetic attention/text mask IoU is0.10–0.24, so these attention regions are not reliable text segmentation.

The legacy comparator, run only after selections froze, favors Controller1 on every family and records regressions versus hybrid for DINO on DarkBay and Outlined. DINO and CLS-only choose different controls on four families, but attention does not consistently improve independent scores or visual results. These are useful contrasting controllers for further research, not a recognition-success result. No new solver was trained or evaluated.

The galleries show original native-size hybrid templates with fresh seeds. This differs from v4's shared recognition-training curriculum, which changed native height, transcripts and font support. No cross-version reader-accuracy comparison is made here.

Controller 1 (v4) remains unchanged. Controller 2 has the same task: use unlabeled target pixels and hybrid starting recipes to improve text/background resemblance. Its underlying model is the official self-supervised DINO ViT-S/8, with frozen cached weights. This is an image-only model: no VLM, language encoder, OCR or target annotations are used.

## Mechanism

RGB images are resized to 256×64 and ImageNet-normalized. The model yields an 8×32 patch grid, 384-dimensional patch embeddings and six final-layer CLS attention heads. Each attention head is scaled by its own 10th/90th percentiles. Each head supplies attended and complementary pooled patch embeddings. The comparator also uses CLS, spatial attention occupancy and attention-weighted RGB histograms. Attention is saliency, not a calibrated text mask; it can select clutter. No custom CNN or U-Net from Controller 1 guides this search.

Fixed score weights: global .20, focus .35, complement .25, attention layout .10, region colors .10. Distribution energy distances are normalized against fit target variability and the hybrid starting distribution. Same image path for real and synthetic comparisons. Synthetic completeness and usability checks guard candidate validity; known synthetic text masks are used only after selection for attention diagnostics.

## Search and evidence

The existing renderer exposes palettes, the existing 96-font bank, geometry, length 2–8 and all sampled effects. Palette proposals use DINO-attended/complementary colors from fit pixels. Stages search palettes, fonts, geometry, effects and joint variations. No family-specific font, charset, answer length, background constants or external target-library assets are injected.

Eight common fit recipes and sixteen distinct selection recipes are used per family. The shortlist combines 32 diverse fit candidates with eight CLS leaders. Up to 16 unique quality-constrained profiles are retained; an archive is never padded. A CLS-only selector uses the same candidate pool, which was generated under the attention objective, so this is a selection ablation rather than a fully independent search ablation.

Selected controls are frozen before diagnostic test scoring. All four comparison arms use the same 24 original hybrid templates and fresh matched seeds: neutral hybrid, frozen Controller 1 profile, DINO, and CLS-only. A length adjustment can change a synthetic string, which is recorded in generated labels. Targets use the first up-to-64 images from previously exposed v4 evaluation pixel manifests. No answer manifests are opened. These are reused diagnostic images, not a new held-out/generalization claim; previous answer exposure elsewhere in the project is disclosed. Controller 1 had a different search budget, so this is not an isolated backbone-only experiment.

## Run

Use the existing project Python environment. `checks5.py` verifies extraction against official attention, order invariance, frozen weights and zero identical-distribution distance. `build5.py` fits all five families and writes profiles, histories, image samples and diagnostics. It skips completed families on resume. `present5.py` creates phone-readable comparison and attention galleries plus `report.html`.

The vendor files retain the official Apache-2.0 license. The sole transformer source edit makes its `utils` import package-relative. Source and weight hashes are recorded in `protocol.json`. The renderer and hybrid remain external dependencies at their existing project paths.

`sample5.py --family City --count 128 --output <new-directory>` exports fresh synthetic images and masks with assigned labels from the retained DINO profiles. It cycles over profile slots and retries failed renderer-validity checks with fresh seeds, failing after20 attempts instead of silently replacing a profile. Existing output directories are refused. The checked `export_check/` contains one City sample from each of the16 profiles.

This MVP builds and visually evaluates a second controller. No new reader training or downstream accuracy improvement is claimed. V4's frozen reader results remain a separate experiment.
