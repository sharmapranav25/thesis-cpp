# Curriculum Learning Literature Notes — for M3 design

Compiled 2026-05-05 from a deep-dive scan triggered by M3 plateauing at 21.5% synth val (run 5) with a clear self-paced-learning failure mode. Each entry: what it is, where to find it, what it contributes, and how it maps to our M3 problem.

---

## 1. Foundational papers

### Curriculum Learning (Bengio et al, ICML 2009)
- The canonical paper introducing curriculum learning to deep nets — train easy first, hard later. Inspired by Elman 1993.
- Predefined CL: humans pick easy-to-hard ordering.
- M2 maps loosely to this idea but uses an external teacher to define "easy."

### Self-Paced Learning for Latent Variable Models (Kumar, Packer, Koller, NeurIPS 2010)
- URL: https://ai.stanford.edu/~koller/Papers/Kumar+al:NIPS10.pdf
- Foundational SPL paper. Adds regularizer `g(v;λ)` to loss; binary `vᵢ ∈ {0,1}` selects samples; threshold on per-sample loss.
- **As λ↑, more samples included.** "Easy" defined by *current model's* loss, not external.
- **Failure mode (key for us):** at random init, model's loss is essentially noise → "easy" set is arbitrary → SPL concentrates on noise. This is run 4-5's exact problem.
- Quote: *"Self-paced learning measures the difficulty of training samples based on the performance of the trained model, so the order of samples changes according to model feedback."*

---

## 2. SPL extensions (most directly relevant to M3)

### Self-Paced Learning with Diversity (SPLD) — Jiang, Meng, Yu, Lan, Shan, Hauptmann, NeurIPS 2014
- URL: https://papers.nips.cc/paper/5568-self-paced-learning-with-diversity
- Project page: http://www.lujiang.info/spld/index.html
- **The literature's named fix for run 5's failure**: SPL alone overfits to a homogeneous easy-set; SPLD adds a diversity regularizer.
- Mechanism: pre-cluster training data; sample-selection objective enforces selecting easy samples *while spanning clusters*.
- Optimization is non-convex but solvable in linearithmic time.
- Quote: *"An effective SPL strategy should generate a curriculum that includes easy and diverse examples that are sufficiently dissimilar from what a DNN has already learned."*
- **For M3:** if we keep "no teacher," SPLD is the right next step. Pre-cluster the 95k synth pool (e.g., k-means on encoder features after a warmup pass, or PHP-renderer-knob clusters), constrain trivial-bucket draws to span clusters.

### Self-Paced Curriculum Learning (SPCL) — Jiang, Meng, Zhao, Shan, Hauptmann, AAAI 2015
- URL: https://cdn.aaai.org/ojs/9608/9608-13-13136-1-2-20201228.pdf
- ResearchGate: https://www.researchgate.net/publication/279853657_Self-paced_Curriculum_Learning
- **The literature-named hybrid** combining predefined curriculum (instructor-driven prior) with self-paced learning. Optimization subject to BOTH the SPL inequality AND a predefined-difficulty inequality.
- Early training dominated by instructor prior; SPL term dominates as model strengthens.
- **For M3:** this is exactly the M3-hybrid (M1.5 buckets cycles 0-3, student buckets cycles 4-14). Has a 2015 citation we can point at.
- Implementation cost: low — read M1.5 buckets for cycles 0-3, swap sampler at cycle 4.

### MentorNet — Jiang, Zhou, Leung, Li, Fei-Fei, ICML 2018
- URL: https://proceedings.mlr.press/v80/jiang18c/jiang18c.pdf
- arXiv: https://arxiv.org/abs/1712.05055
- Code: https://github.com/google/mentornet
- A *separate* small network learns to weight the StudentNet's per-sample loss. Trained on a small clean validation set to predict "is this sample worth training on?"
- **20% absolute accuracy gain over baseline on CIFAR-10 with 40% noise.** State-of-art on WebVision.
- Crucially: MentorNet *adapts during training* — not a frozen teacher like M1.5. Could be trained on the same synth pool without external labels.
- **For M3:** more complex than SPCL/SPLD but most powerful. A learned per-sample weighter could capture difficulty signals neither EMA nor M1.5 NLL provide.

---

## 3. Teacher-driven adaptive curricula

### Teacher-Student Curriculum Learning (TSCL) — Matiisen, Oliver, Cohen, Schulman, 2017
- arXiv: https://arxiv.org/abs/1707.00183
- Bandit-based meta-learner. Tasks (or buckets) are arms; reward = **slope of student's learning curve** on that arm.
- Variants: Online Naive, Naive, Window, Sampling, Boltzmann.
- **Insight that matters for M3:** instantaneous loss is a noisy signal; *learning progress* (Δ-accuracy) is more robust.
- Validated on LSTM decimal addition + Minecraft maze navigation.
- **For M3:** replace fixed phase weights {15/25/60} with bandit-selected weights. Reward = val_acc improvement on samples drawn from each bucket between consecutive cycles.

### Curriculum Learning for RL Domains (Narvekar et al, JMLR 2020)
- URL: https://jmlr.org/papers/volume21/20-212/20-212.pdf
- Survey of RL-flavored curriculum methods. Lots of bandit/POMDP framings of teacher selection.

---

## 4. Sample-importance / per-sample scoring

### An Empirical Study of Example Forgetting (Toneva, Sordoni, Combes, Trischler, Bengio, Gordon, ICLR 2019)
- arXiv: https://arxiv.org/abs/1812.05159
- Code: https://github.com/mtoneva/example_forgetting
- Defines "forgetting event": sample transitions correct→incorrect during training.
- **Key finding:** unforgettable examples can often be removed entirely without hurting test acc. Forgotten examples are the important ones.
- Forgetting score = number of forgetting events per sample over training.
- **For M3:** replace per-sample loss EMA with per-sample forgetting count. More temporally robust signal — captures whether the model truly knows a sample, not just got it right once.

### Deep Learning on a Data Diet (Paul, Ganguli, Dziugaite, NeurIPS 2021)
- arXiv: https://arxiv.org/abs/2107.07075
- Two scoring methods:
  - **GraNd** = gradient norm at sample
  - **EL2N** = L2 norm of (predicted softmax − one-hot label)
- Computable in **first few epochs**, well-approximates full-training importance.
- Can prune 50% of CIFAR-10 / 25% of CIFAR-100 with no test-acc loss.
- **Critical warning that explains run 5:** *"Focusing only on the highest error examples using GraNd or EL2N scores leads to bad coverage of the data distribution, as it likely excludes entire subpopulations present in the test data, fitting only small numbers of very difficult examples rather than maintaining a variety needed for good test error."*
- **For M3:** EL2N is essentially what our `_last_per_sample_loss` is approximating via AR forward. Bucketing by EL2N + diversity (SPLD-style) would be more principled than instantaneous loss EMA.

---

## 5. Loss-embedded curriculum (CurricularFace family)

### CurricularFace — Huang, Wang, Tao, He, Wei, Wang, Zhang, CVPR 2020
- arXiv: https://arxiv.org/abs/2004.00288
- Embeds curriculum in the loss function rather than as sample selection.
- Adaptively reweights easy/hard samples by training stage (early = easy emphasis, late = hard emphasis).
- Mechanism in face-rec: angular margin loss with adaptive modulation function.
- **For M3:** an alternative to bucket-based sampling — instead reweight per-sample losses based on training progress. Less intrusive on the data pipeline.

### HeadPose-Softmax (2023) — extends CurricularFace
- URL: https://www.sciencedirect.com/science/article/abs/pii/S0031320323002522
- Adds head-pose adaptive curriculum. Not directly relevant but shows the family is alive.

---

## 6. Competence-based curriculum

### Competence-based Curriculum Learning for NMT (Platanios, Stretcu, Neubig, Poczos, Mitchell, NAACL 2019)
- arXiv: https://arxiv.org/abs/1903.09848
- Code: https://github.com/josephch405/curriculum-nmt
- **Pacing function = function of model competence, not training step.**
- `c(t) = √(t·(1−c₀²)/T + c₀²)` — square-root pacing, c₀ ≈ 0.01.
- At step t, sample only from cumulative-difficulty fraction ≤ c(t).
- **70% training time savings + 2.2 BLEU gain on NMT.**
- Difficulty metrics for NMT: sentence length, word rarity (predefined).
- **For M3:** the principle of competence-based pacing applies. We could compute a "competence" score per cycle (e.g., from val_acc) and use it to gate which difficulty quantile is currently visible.

---

## 7. STR-specific curriculum

### TEACH: Text Encoding as Curriculum Hints (2025)
- arXiv: https://arxiv.org/abs/2508.01153
- HTML: https://arxiv.org/html/2508.01153
- Injects **ground-truth text as auxiliary input** early in training, progressively masks it via loss-aware masking.
- Transitions from label-guided to vision-dominant recognition.
- **For M3:** parallel to teacher-forcing-as-warmup. Could feed GT label as hint feature for cycles 0-3, then mask it.

### Curriculum Learning for Scene Text Recognition (Yan et al, JEI 2021)
- URL: https://www.spiedigitallibrary.org/journals/journal-of-electronic-imaging/volume-30/issue-4/043006/Curriculum-learning-for-scene-text-recognition/10.1117/1.JEI.30.4.043006.short
- Two-perspective difficulty: human (visual complexity) and machine (model loss).
- **Validated that two-stage learning (warmup + full mixed) outperforms pure curriculum or pure mixed.** Direct support for SPCL-style hybrids.

### Instruction-Guided Scene Text Recognition (2024)
- arXiv: https://arxiv.org/html/2401.17851
- Less directly curriculum; more about instruction conditioning.

---

## 8. Surveys

### A Survey on Curriculum Learning (Wang, Chen, Zhu, 2021)
- arXiv: https://arxiv.org/abs/2010.13166
- Tsinghua: https://mn.cs.tsinghua.edu.cn/xinwang/PDF/papers/2021_A%20Survey%20on%20Curriculum%20Learning.pdf
- **Framework: Difficulty Measurer + Training Scheduler.**
- Taxonomy: predefined CL vs automatic CL (SPL, transfer teacher, RL teacher, other).
- M2 = transfer teacher. M3 = SPL.

### Curriculum Learning: A Survey (Soviany, Ionescu, Rota, Sebe, 2021)
- arXiv: https://arxiv.org/pdf/2101.10382
- Four broad CL categories: data, model, task, target.
- Comprehensive failure-mode catalog.

---

## 9. Self-paced supporting work

### Self-Paced Resistance Learning Against Overfitting on Noisy Labels (2021)
- arXiv: https://arxiv.org/abs/2105.03059
- Adds resistance loss + memorization-effect-driven sample selection.
- Shows DNNs first memorize easy samples, then hard, then noisy → SPL alone amplifies the corruption.

### Self-Paced Learning: An Implicit Regularization Perspective (Fan et al, AAAI 2017)
- URL: https://ojs.aaai.org/index.php/AAAI/article/view/10809/10668
- Theoretical: SPL as implicit regularization. Why it sometimes works.

### Self-Adaptive Training (Huang, Qu, Wang, Yang, NeurIPS 2020)
- Beyond ERM. Trains with self-distillation soft labels generated by the model's own running predictions.
- Robust to label noise without external teacher.

---

## 10. Continual learning / catastrophic forgetting

### Overcoming Catastrophic Forgetting in Neural Networks (Kirkpatrick et al, PNAS 2017)
- URL: https://www.pnas.org/doi/10.1073/pnas.1611835114
- Elastic Weight Consolidation. Not curriculum per se but the forgetting analysis is relevant — when buckets shift mid-training, we incur forgetting on samples that left the active distribution.

### The Effect of Task Ordering in Continual Learning
- arXiv: https://arxiv.org/pdf/2205.13323
- Shows ordering directly affects forgetting; right curriculum can mitigate it.

---

## 11. Ranked summary — what to actually try for M3

| Option | Method | Citation | Effort | Expected payoff |
|---|---|---|---|---|
| **A** | SPCL hybrid (M1.5 buckets cycles 0-3, student buckets cycles 4+) | Jiang 2015 AAAI | Low | High — direct fix for cold-start noise |
| **B** | SPLD (cluster diversity in trivial bucket) | Jiang 2014 NeurIPS | Medium | High — fixes run 5 overfit, preserves "no teacher" |
| **C** | Forgetting-event signal instead of EMA | Toneva 2019 ICLR | Low | Medium — more stable signal |
| **D** | Bandit (TSCL) with learning-progress reward | Matiisen 2017 | Medium | Medium — adaptive weights, but added meta-state |
| **E** | EL2N scores (one-shot warmup pass, then static) | Paul 2021 NeurIPS | Low | Medium — simpler than EMA |
| **F** | MentorNet (learned per-sample weighter) | Jiang 2018 ICML | High | Highest theoretical, hardest to implement |
| **G** | Loss-embedded (CurricularFace style) | Huang 2020 CVPR | Medium | Unclear for STR |
| **H** | Competence pacing (c(t) gating) | Platanios 2019 NAACL | Low | Medium — simple addition |

Top recommendation: **A (SPCL)** + later **B (SPLD)** if we want to keep the "no teacher" axis pure.
