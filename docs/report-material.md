# Report material — CW1 Diabetic Retinopathy Stage Detection

**For Mohamed to write from.** This is not report prose. It is the structure, the figures worth their space, and every number with the file it came from, so that each sentence you write can be checked against a saved run before it goes in.

Two rules that should survive to the final draft:

1. **If you cannot point at the file it came from, don't write it.** Every number below names its source.
2. **Validation numbers and test numbers are different things.** Validation (523 images) chose the model; test (525 images) measured it, once. Never quote one as the other.

---

## 1. Page budget (limit: 20 pages)

| Section | Pages | Figures |
|---|---|---|
| Title, abstract, contents | 1 | — |
| 1. Problem, objectives, system overview | 1.5 | fig35 |
| 2. The dataset and its traps | 2.5 | fig01, fig12, (fig09) |
| 3. Preprocessing and enhancement | 2 | fig13, fig14, (fig18) |
| 4. Augmentation and class balance | 1.5 | fig20, fig21 |
| 5. Model and transfer learning | 2 | fig36 + hyperparameter table |
| 6. Training and experiments | 2.5 | fig22, fig24, (fig23) |
| 7. Evaluation and error analysis | 3.5 | fig26, fig27, fig28, fig31, fig33, (fig30) |
| 8. Prototype and video | 1.5 | 2 app screenshots + the video URL |
| 9. Discussion, limitations, ethics, future work | 1.5 | — |
| References | 0.5 | — |

Figures in brackets go in only if the pages allow. At roughly a third of a page each, the 17 "must" figures take about 6 pages, which leaves 13 for text — enough if the writing stays tight.

---

## 2. Figure shortlist

**Must go in (17):**

| Figure | File | What it shows | Why it earns the space |
|---|---|---|---|
| 35 | `report_figures/fig35_system_overview.png` | The four notebooks, the prototype, the leakage controls | One picture answers "what did you build?" |
| 01 | `results/notebook1_v5/` | Images per stage | The imbalance every later decision reacts to |
| 12 | `results/notebook2_v1/` | Stage mix per camera + what the camera alone scores | The central trap of this dataset |
| 13 | `results/notebook2_v1/` | Preprocessing, step by step on one photo | Rubric row 2 wants the pipeline visible |
| 14 | `results/notebook2_v1/` | Variants A/B/C for each stage | Shows what enhancement actually does |
| 20 | `results/notebook3/figures/` | One image, eight augmented versions | Rubric row 3 |
| 21 | `results/notebook3/figures/` | What balancing does to one epoch | Rubric row 3 |
| 36 | `report_figures/fig36_model_architecture.png` | The model, layer by layer, with parameter counts | Rubric row 4 |
| 22 | `results/notebook3/figures/` | Every configuration against both baselines | The experiment result in one chart |
| 24 | `results/notebook3/figures/` | Accuracy and loss curves, both phases | Required by the brief |
| 26 | `results/notebook4/figures/` | Test confusion matrix | Required by the brief |
| 27 | `results/notebook4/figures/` | Per-stage precision, recall, F1 with intervals | Required by the brief, and shows the uncertainty |
| 28 | `results/notebook4/figures/` | ROC curve and the yes/no table | The screening result |
| 31 | `results/notebook4/figures/` | Model vs camera-only, camera by camera | Your best evidence against shortcut learning |
| 33 | `results/notebook4/figures/` | Grad-CAM, one correct example per stage | Rubric rows 6 and 9 |
| — | your screenshots | App: answer + probabilities; app: Grad-CAM | Rubric row 8 |

**If space allows:** fig09 (one photograph labelled 2, 3 and 4 — label noise), fig18 (the camera is still recognisable after preprocessing), fig23 (frozen vs fine-tuned), fig30 (how far off the mistakes are), fig34 (Grad-CAM on mistakes), fig19 (the uneven cut-off flaw), fig25 (validation confusion matrix).

**Leave out:** everything else. A figure with no sentence pointing at it is wasted space.

---

## 3. Section by section

### Section 1 — Problem and objectives (1.5 pages)

**Argue:** what diabetic retinopathy is, why screening matters, what the system does (two answers: DR yes/no, then the stage), and what it is not (not a diagnostic device).

**Material:**
- The five stages, in one table: 0 No DR; 1 Mild (microaneurysms only); 2 Moderate; 3 Severe; 4 Proliferative (new fragile vessels). Source: ICDR scale, Wilkinson et al. 2003 — cite it.
- The two-step rule, in one sentence: *DR is present if the probabilities of stages 1–4 add up to 0.5 or more; the stage is then the most likely of stages 1–4.* Source: `best_config.json`, field `decision_rule`.
- Figure 35 for the system overview.

**Trap:** don't write that the system "diagnoses" or "screens patients". It classifies photographs.

### Section 2 — The dataset and its traps (2.5 pages)

**Argue:** what APTOS 2019 is, what is wrong with it, and what you did about each problem *before* modelling.

**Material (all from `dataset_summary.json` / notebook 1 figures):**
- 3,662 photographs, one clinician's grade each, 17 different image sizes.
- Stages: 1,805 / 370 / 999 / 193 / 295 = 49.3 / 10.1 / 27.3 / 5.3 / 8.1 %.
- **Duplicates:** 251 byte-identical files; 96 extra copies removed; 62 identical files whose copies carried *different* stages — dropped. Near-duplicates found with a correlation fingerprint: 11 pairs in 9 groups; 7 groups (15 images) had conflicting stages and were dropped. Total dropped 173, kept 3,489.
- **Why the threshold is 0.982:** the sweep found no chain-free cut below 0.97, and the cut was placed in the middle of the widest empty band (0.9726–0.9904) of the best-match curve. Four earlier attempts failed — worth two sentences, because the marker sees method, not luck (the version table is in the results log).
- **Split:** stratified 70/15/15 with duplicate groups kept together → 2,441 / 523 / 525.
- **The camera trap:** stage 0 is 93.1 % of one camera's photographs and 0.5 % of another's. A model that only knows the image size scores macro-F1 0.326 and gets DR yes/no right 88.9 % of the time on validation — without ever looking at a retina. Source: `baselines_val.csv`, fig12.
- **Label noise:** fig09 shows one photograph labelled 2, 3 and 4 in three copies.

**Trap:** don't claim the dataset is now clean. Say what remains: the method only finds copies of the same photograph, APTOS has no patient IDs, so the same eye photographed twice can still sit in two splits.

### Section 3 — Preprocessing and enhancement (2 pages)

**Argue:** every photograph is given the same field of view *because* the camera hints at the diagnosis; then the three variants, and why the choice between them was measured, not assumed.

**Material:**
- Steps: find the retina by thresholding the dark background → crop to a square → resize to 512 px with area interpolation (this is also the noise removal) → apply one common window: full retina width, top and bottom trimmed flat at 0.74 × radius. Source: `dr_preprocessing.py`, fig13.
- **Why that window:** some cameras deliver a full circle, others cut the retina off, and the shape alone nearly separates DR from no DR (full-circle cameras are 93–100 % stage 0). Giving every photo the same shape removes that cue.
- Nothing was cut off more than the window allows (0 photographs); median share of the visible retina kept 0.831; 2 photographs had to be enlarged. Source: `preprocessing_summary.json`.
- Variant B (CLAHE on the LAB lightness channel, clip 2.0, 8×8 tiles, then an unsharp mask) raises median contrast from 15.9 to 23.2 and median sharpness from 24.0 to 612.1. Say plainly that part of that sharpness rise is amplified noise — the Laplacian cannot tell an edge from a speck.
- Variant C (Ben Graham) evens out lighting: contrast 16.7, sharpness 363.7.
- **Preprocessing does not remove the camera:** a random forest still recognises which camera took a photograph from seven colour and texture numbers, 93 % (A), 93 % (B), 85 % (C) against a 32.5 % chance level. Source: `preprocessing_summary.json`, fig18. This is why section 7 reports results camera by camera.

**Trap:** don't say CLAHE "improves image quality". It increases measured contrast; whether that helps the model is experiment E2, and the answer there is "a little".

### Section 4 — Augmentation and class balance (1.5 pages)

**Argue:** which transforms are realistic for fundus photographs, why colour shifts and cutout were left out, and that augmentation only ever touches the training split.

**Material:**
- Flips (both axes), rotation at any angle, zoom ±10 %, shift ±5 %, brightness and contrast ±15 %. All as Keras layers inside the model, active only in training. Sources: `dr_training.py` (`make_augmenter`), fig20.
- Left out on purpose: hue shifts (lesion colour carries meaning; Graham found colour augmentation did not help) and cutout (it could erase the only lesion in a mild case).
- Balancing options compared in E3: none, class weights (`n / (5 × n_stage)` from the training split only), oversampling to a fifth per stage. fig21.
- Result: neither cleared the seed spread on macro-F1, so **no balancing** was used. But class weights gave the highest DR sensitivity of any run (0.974) at the cost of 4 points of accuracy. Source: `experiment_summary.csv`.

**Trap:** the class weights come from the training labels only — say so; it is the kind of leakage a marker looks for.

### Section 5 — Model and transfer learning (2 pages)

**Argue:** why EfficientNetB3 with ImageNet weights, what was frozen when, and why the head is deliberately small.

**Material (fig36 carries most of it):**
- EfficientNetB3 at 300 px: 10,791,220 parameters, 383 layers, last convolution block 10 × 10 × 1536.
- Head: global average pooling → dropout 0.3 → dense 5 with softmax = 7,685 parameters.
- Phase 1: backbone frozen, only those 7,685 weights train, Adam at 1e-3, up to 8 epochs.
- Phase 2: top 25 % of the layers unfrozen — **which is 70 % of the weights (7,562,791)**, because parameters concentrate near the top — Adam at 1e-4, up to 20 epochs.
- BatchNorm layers (78 of them) stay frozen throughout, so their ImageNet statistics survive small batches. This is the official Keras advice for fine-tuning — cite the guide.
- EfficientNet takes pixels as 0–255 because it rescales internally. Feeding it 0–1 images is a real and common bug; worth one sentence.
- Model selection watches **validation macro-F1**, not accuracy, because accuracy rewards ignoring the rare stages — the trap the lecturer's own notebook fell into.

**Hyperparameter table** (put it here, it answers rubric row 4 directly): input 300 px; batch 16; head lr 1e-3; fine-tune lr 1e-4 (1e-5 tested, worse — E4); unfrozen share 25 % of layers; dropout 0.3; early stopping patience 3 then 5 on validation macro-F1; ReduceLROnPlateau halving after 2 flat epochs; seeds 42 and 43.

### Section 6 — Training and experiments (2.5 pages)

**Argue:** the experiments were planned, one thing changed at a time, each judged on validation only, with two seeds so that noise is not mistaken for a result.

**Material — the experiment table (`experiment_summary.csv`):**

| Config | Macro-F1 (min–max) | QWK | Accuracy | DR accuracy |
|---|---|---|---|---|
| Camera only (no retina) | 0.326 | 0.602 | 0.702 | 0.889 |
| E1 B0 + A | 0.600 (0.597–0.604) | 0.854 | 0.811 | 0.972 |
| E2 B0 + B | 0.617 (0.611–0.623) | 0.849 | 0.823 | 0.971 |
| E2 B0 + C | 0.589 (0.584–0.595) | 0.852 | 0.802 | 0.958 |
| E3 + class weights | 0.620 (0.613–0.627) | 0.834 | 0.780 | 0.960 |
| E3 + oversampling | 0.615 (0.602–0.627) | 0.844 | 0.784 | 0.967 |
| E5 MobileNetV2 | 0.600 (0.593–0.606) | 0.835 | 0.810 | 0.966 |
| **E5 EfficientNetB3** (1 run) | **0.641** | **0.867** | **0.830** | 0.967 |
| E4 same at lr 1e-5 (1 run) | 0.586 | 0.802 | 0.799 | 0.931 |

**What to say about each:**
- **E2:** B beats A by 0.016 against a seed spread of 0.012 — a win by the rule fixed in advance, but a narrow one; on the mixed camera the gap is much clearer (0.705 vs 0.608). C scored *below* A, so Ben Graham was rejected on evidence.
- **E3:** neither method cleared the spread, so none was kept. Report the sensitivity/accuracy trade-off.
- **E4 (fine-tuning):** +0.07 to +0.21 macro-F1 in every single run — the largest effect in the project. fig23 if space.
- **E4 (learning rate):** 1e-5 scored 0.586 against 0.641 at 1e-4, so the rate was chosen by measurement.
- **E5:** MobileNetV2 fell below B0; B3 won by 0.024. **State the caveat**: B3 ran with one seed, so its own spread is unknown and the comparison borrows the other runs' spread.
- 15 runs, 86 minutes of GPU training in total, each run in its own process because TensorFlow left about 0.3 GB behind per run.

**Overfitting (fig24):** at the kept epoch (21 of 26) training accuracy was 0.909 against validation 0.826, and training loss 0.241 against validation 0.618. Validation loss bottomed at epoch 12 and rose afterwards while validation macro-F1 kept improving to epoch 21 — which is exactly why the checkpoint follows macro-F1. Early stopping with best-weight restore ended the run.

### Section 7 — Evaluation and error analysis (3.5 pages — the heaviest section)

**Argue:** the test split was used once; the model is strong at detection and weak at grading; and it is not exploiting the camera.

**Headline table (`test_metrics.json`):**

| | Model (test) | 95 % interval | Camera only | Always "No DR" |
|---|---|---|---|---|
| Accuracy | 0.817 | 0.783–0.848 | 0.697 | 0.514 |
| Macro-F1 | 0.617 | 0.559–0.673 | 0.379 | 0.136 |
| QWK | 0.900 | 0.876–0.922 | 0.599 | 0.000 |
| DR yes/no accuracy | 0.973 | 0.958–0.985 | 0.865 | 0.514 |
| DR sensitivity | 0.953 | 0.925–0.976 | 0.749 | 0.000 |
| DR specificity | 0.993 | 0.981–1.000 | 0.974 | 1.000 |
| DR ROC-AUC | 0.994 | 0.986–0.999 | — | — |
| Referable DR (≥2) sens / spec | 0.931 / 0.919 | — | 0.726 / 0.866 | 0 / 1 |

- Say that the model reproduced **100 %** of its validation answers before the test images were touched — the check that the pipeline is the same one.
- Validation → test drift: macro-F1 −0.024, accuracy −0.013, QWK +0.033, sensitivity +0.016. Small and in both directions, so selection on validation did not inflate the result.
- In screening language: **12 of 255 diseased eyes missed, 2 of 270 healthy eyes flagged.**
- Explain the bootstrap in one sentence: resample the 525 test answers with replacement 2,000 times, take the middle 95 % of each metric.

**Per stage (`test_per_stage.csv`, fig27):** F1 = 0.975 / 0.475 / 0.748 / 0.286 / 0.603 for stages 0–4, with training images 1,257 / 236 / 644 / 119 / 185. F1 follows the training count almost exactly. **Always quote Severe's interval (0.11–0.47)** — with 26 test images the number on its own is not meaningful.

**Errors (fig26, fig30):** 96 mistakes in 525; 72 of them (75 %) one stage away, which is why QWK is 0.90 while macro-F1 is 0.62. The mistakes run downwards: 14 of 26 Severe and 12 of 39 Proliferative eyes were called Moderate, and 25 of 51 Mild eyes were called Moderate.

**Camera by camera (fig31, `test_by_camera.csv`) — write this carefully, it is the best part:**
- On the 2588×1958 camera (83 test photographs, 36 % healthy) the model reaches 0.819 accuracy, macro-F1 0.722, DR sensitivity 0.981, specificity 1.000. The camera-only model calls every one of those photographs "No DR": accuracy 0.361, sensitivity 0.
- On cameras that photographed only one stage, the camera-only model ties the yes/no question trivially by predicting that stage.
- Conclusion to draw: **the model's advantage appears exactly where the camera carries no information**, which is the honest answer to "is it just recognising the camera?".

**Image quality (`test_by_quality.csv`):** DR yes/no accuracy is stable across sharpness thirds (0.954 / 0.983 / 0.983), but errors of two or more stages are six times more common in the blurriest third (10.3 % vs 1.7 %). Note the confound: the sharpest third is also 81 % healthy, so stage accuracy across thirds cannot be read as a quality effect.

**Grad-CAM (fig33, fig34):** across all 525 test images the median share of heat inside the retina window is **0.961**, against **0.601** for a map spread evenly — the model looks at the retina, not the border or the black background. Say what Grad-CAM cannot do: the maps are 10 × 10, each cell about 30 × 30 pixels of the input, so they show a region, not a lesion, and a plausible map does not make an answer correct.

### Section 8 — Prototype and video (1.5 pages)

**Material:**
- What it does: upload or pick a photograph → the retina cropped to the same window used in training → the enhancement variant → DR yes/no with its probability → the stage with all five probabilities → Grad-CAM → quality warnings → disclaimer.
- Runs on the Mac (Python 3.13.9, TensorFlow 2.21.0, Keras 3.15.1) with the model trained on Kaggle (TensorFlow 2.20.0, Keras 3.13.2); about 0.7 s per photograph on the CPU.
- One shared preprocessing file for Kaggle and the app, so the app cannot drift from training — this is worth a sentence, it is a design decision a marker will notice.
- Quality warnings fire outside the 1st–99th percentile of the training photographs; the middle-90 % range was tried first and flagged 26 % of the training photographs themselves, which is why the wider range is used.
- Two screenshots only: the answer with the probability chart, and the Grad-CAM panel.
- The video URL goes here, and in the abstract.

### Section 9 — Discussion, limitations, ethics, future work (1.5 pages)

**The honest summary sentence:** *the system detects diabetic retinopathy reliably (sensitivity 0.95, specificity 0.99, AUC 0.994) and grades its severity unreliably, especially at the severe end (Severe F1 0.29).* Everything in that sentence has a figure behind it.

**Limitations — write them as a list, they earn marks:**
1. One dataset, one region (Aravind Eye Hospital, India), a handful of cameras. Nothing here shows the model would transfer to another population or another camera.
2. Labels come from a single clinician per image and the organisers warn they are noisy; 77 of the images we removed carried contradictory stages between copies.
3. APTOS has no patient IDs, so the same eye photographed twice could sit in two splits. We can only detect copies of the same photograph.
4. Severe has 119 training and 26 test images; its F1 interval (0.11–0.47) is too wide to call.
5. EfficientNetB3 was chosen on a single seed.
6. The preprocessing does not remove the camera's signature (85–93 % recognisable), so some camera information remains available to the model.
7. Grad-CAM shows where, not why, and at 30-pixel resolution.
8. The test set was used once, which is correct, but it also means the reported numbers have no second confirmation.

**Ethics and impact — distinguish three things explicitly:**
- *What the experiment shows:* on 525 held-out APTOS photographs, these numbers.
- *What a real system might do:* flag likely disease for a specialist to review, in places where specialists are scarce.
- *What would be needed first:* prospective clinical validation, several populations and cameras, regulatory approval, a defined role for the clinician, and a plan for what happens when it is wrong. A missed severe case is the harm that matters here, and our model under-grades severity.

**Future work (grounded in what we measured, not generic):** train on EyePACS or IDRiD to test transfer; try class weights *with* EfficientNetB3 (weights raised sensitivity on B0, but we never combined them); a second seed for B3; ordinal regression or a stage-aware loss, since three quarters of the errors are one stage away; and an explicit "image not gradable" output, since large errors concentrate in the blurriest third.

---

## 4. The numbers you will quote most, with sources

| Number | Value | File |
|---|---|---|
| Images kept / split | 3,489 → 2,441 / 523 / 525 | `dataset_summary.json` |
| Camera-only baseline (validation) | macro-F1 0.326, DR acc 0.889 | `baselines_val.csv` |
| Chosen model | EfficientNetB3, 300 px, variant B, no balancing | `best_config.json` |
| Validation macro-F1 / QWK | 0.641 / 0.867 | `best_config.json` |
| Test accuracy / macro-F1 / QWK | 0.817 / 0.617 / 0.900 | `test_metrics.json` |
| Test DR sensitivity / specificity / AUC | 0.953 / 0.993 / 0.994 | `test_metrics.json` |
| Missed / false alarms | 12 of 255 / 2 of 270 | fig26, fig28 |
| Per-stage F1 | 0.975 / 0.475 / 0.748 / 0.286 / 0.603 | `test_per_stage.csv` |
| Mistakes one stage away | 72 of 96 (75 %) | `test_metrics.json` |
| Mixed camera, model vs camera-only | 0.819 vs 0.361 accuracy | `test_by_camera.csv` |
| Grad-CAM heat inside the retina | 0.961 (even map: 0.601) | `test_metrics.json` |
| Parameters | 10,791,220 total; 7,685 then 7,562,791 trainable | measured with `dr_training.build_model` |

---

## 5. What must not go in

- No claim that the system diagnoses, screens, or is safe for clinical use.
- No accuracy figure quoted without saying which split it came from.
- No per-stage number for Severe without its interval.
- No claim that CLAHE "improves accuracy" in general — in E2 it gained 0.016 macro-F1 against a 0.012 seed spread, in this setting, with this backbone.
- No claim that EfficientNetB3 is better than EfficientNetB0 — one seed.
- No screenshot without a sentence that uses it.
- Nothing invented: if a number is not in a saved run, it does not exist.

---

## 6. Rubric cross-check

| Requirement | Where it is evidenced |
|---|---|
| DR explained, dataset described, ethics | §1, §2, §9; fig01, fig12 |
| Preprocessing: contrast, edge enhancement, resizing, noise | §3; fig13, fig14; `dr_preprocessing.py` |
| Augmentation and class balancing, justified | §4; fig20, fig21; E3 in §6 |
| CNN architecture, transfer learning, hyperparameters | §5; fig36; hyperparameter table; E4, E5 |
| Validation, callbacks, early stopping, overfitting | §6; fig24; `history.csv` |
| Accuracy/loss curves, precision, recall, F1, confusion matrix | §6, §7; fig24, fig26, fig27, fig28 |
| Interpretation and error analysis | §7; fig30, fig31, fig33, fig34 |
| Detects DR **and** the stage | §1 rule; §7 both metric families |
| Commented, original code | GitHub repo link; four notebooks + three shared modules |
| Prototype + hosted video | §8; screenshots + URL |
| Innovation, impact, deployment, ethics, future work | §9 |
