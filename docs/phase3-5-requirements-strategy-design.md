# Phases 3–5 — Requirement Map, Strategy & System Design

**Module:** Computer Vision — CW1 "Diabetic Retinopathy Stage Detection"
**Prepared:** 17 September 2026 · **Deadline:** 3 October 2026
**Builds on:** `claude/phase1-background-and-assignment-analysis.md`

Items marked **[Additional knowledge]** are not in the practicals. Each one will be explained in simple terms before we use it.

---

## 0. Decisions confirmed so far

| Topic | Decision | Reason |
|---|---|---|
| Training compute | **Kaggle Notebooks (GPU)** | Free GPU. The dataset attaches directly, so there is no 10 GB download to your Mac. |
| Prototype machine | **Your MacBook (Apple M2)** | TensorFlow installs on Apple silicon with `pip install tensorflow`. There is no official Mac GPU support, but the CPU easily handles one image at a time. |
| Framework | **TensorFlow / Keras** | See the explanation below. |
| "DR + stage" | **One model, two answers** | See the explanation below. |
| Extra lecture material | None | Classical image processing (contrast, sharpening) is treated as additional knowledge. |
| **Report authorship** | **Mohamed writes the report. Claude supplies material only, and nothing goes in unverified.** | It is his submission; he must be able to stand behind every sentence and every number. |

### Why TensorFlow/Keras (plain English)

A *framework* is the toolbox we write the deep-learning code with. The two big ones are **TensorFlow/Keras** and **PyTorch**. Both could do this project, so we pick the one that helps you most:

1. **It matches your practicals.** Six of your seven image-model practicals, and the lecturer's own DR example, use Keras. The marker will recognise the knowledge, and your lecturer said knowledge must be *witnessed* in the work.
2. **You already know the pieces.** `model.fit`, the callbacks (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau) and the pretrained models (MobileNetV2, EfficientNet) all appear in your notebooks.
3. **It runs in both places.** Kaggle Notebooks come with it, and it installs on your M2 Mac for the Streamlit app.
4. **Alternative — PyTorch.** You used it once, for the augmentation comparison. It needs hand-written training loops and gives no advantage here.

### What "classify DR as well as the stage" means (plain English)

For every eye photo, the system must answer **two questions**:

1. **Does this eye have diabetic retinopathy?** (Yes / No)
2. **If yes, how advanced is it?** (the stage)

The dataset uses the international 0–4 scale:

| Stage | Name | What the doctor sees (simplified) |
|---|---|---|
| 0 | No DR | No damage from diabetes |
| 1 | Mild | Only *microaneurysms* — tiny bulges in the smallest blood vessels |
| 2 | Moderate | More than microaneurysms (small bleeds, yellow fatty deposits, etc.), but not severe |
| 3 | Severe | Many bleeds in all four quarters of the retina, or other serious vessel changes (the "4-2-1 rule") |
| 4 | Proliferative | New, fragile blood vessels growing — the highest risk of vision loss |

**Our design: one model, two answers.** The model looks at a photo and outputs five percentages (one per stage) that add up to 100 %. From those five numbers:

- **Question 1 — DR present?** Add the four DR stages together (Mild + Moderate + Severe + Proliferative). If the total is **50 % or more → Yes**.
- **Question 2 — which stage?** If the answer is Yes, pick the DR stage with the highest percentage. If No, the stage is 0.

*Worked example (made-up numbers, just to show the rule):* No DR 10 %, Mild 15 %, Moderate 60 %, Severe 10 %, Proliferative 5 %.
→ DR total = 90 % → **DR present: Yes** → highest DR stage = **Moderate**.

**Why one model instead of two separate models?**

- Training happens once.
- The two answers can never contradict each other (e.g., "No DR" but "Stage 3").
- It is easier to explain.

In the report we evaluate **both** answers: the 5-stage results and the Yes/No results. We will also check on the validation set that this two-step rule does not hurt stage accuracy compared with simply picking the single highest percentage.

---

## 1. Phase 3 — Requirement map

**Where the knowledge comes from**

- *Existing knowledge* = seen in your practicals (numbers refer to the Phase 1 table: #2 MNIST, #6 augmentation comparison, #7 transfer learning, #8 lecturer's DR notebook + guide, #9 tumour U-Net, #10 Siamese, #11 face-recognition app, #12 LLM app).
- *Additional knowledge* = new material we must learn.

| # | Requirement (brief / rubric) | Marks | Existing knowledge | Additional knowledge | Where it's handled |
|---|---|---|---|---|---|
| R1 | Explain DR and its medical importance | Row 1 (10) | DR guide PDF (#8): 5 grades, lesions | Clinical grading scale (ICDR), screening context | Report §1 |
| R2 | Acquire a relevant Kaggle dataset and justify it | Row 1 | Kaggle API (#7–#10) | Attaching data in a Kaggle Notebook; comparing APTOS vs EyePACS | Phase 6 |
| R3 | Describe classes, distribution and splits | Row 1 | Class counts/charts (#7, #8); stratified split (#8, #10) | Duplicate-aware splitting; patient-level limitation | Phase 6 |
| R4 | Ethics and dataset limitations | Rows 1, 9 | "Assistant, not a doctor" framing (#8 guide) | Dataset bias, label noise, privacy, clinical validation | Report §2, §9 |
| R5 | Contrast adjustment | Row 2 (10) | Ben Graham filter (defined only in #8) | **CLAHE** | Phase 7 |
| R6 | Edge enhancement | Row 2 | — | **Unsharp masking** (sharpening) | Phase 7 |
| R7 | Resizing, normalisation, noise removal | Row 2 | Resize, ÷255, model-specific scaling (#2, #7); Gaussian blur | EfficientNet's 0–255 input rule; choosing a denoising filter | Phase 7 |
| R8 | Reproducible, leakage-free preprocessing | Row 2 | Train-only augmentation (#6, #7) | Per-image deterministic steps; one shared preprocessing file for training and app | Phase 5 design, Phase 7 |
| R9 | Justified augmentation | Row 3 (10) | Flips, rotation, zoom, shift, brightness (#6–#8) | Fundus-specific choices; `tf.data` + Keras preprocessing layers | Phase 8 |
| R10 | Class balancing | Row 3 | Class weights (#8) | Oversampling inside the training set; focal loss; macro metrics | Phases 8, 11 |
| R11 | Suitable, justified CNN architecture | Row 4 (20) | CNN basics (slides, #2); MobileNetV2 (#7); EfficientNetB3 (#8) | EfficientNet compound scaling; comparing backbones fairly | Phases 9, 11 |
| R12 | Correct, optimised transfer learning | Row 4 | Freeze → fine-tune (#7, #8) | BatchNorm kept in inference mode; correct input range | Phase 9 |
| R13 | Hyperparameter tuning | Row 4 | Config classes (#7, #8, #10) | Small, planned search on the validation set only | Phase 11 |
| R14 | Validation, callbacks, early stopping, LR scheduling | Row 5 (10) | EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger (#7); cosine LR (#6) | Custom callback that logs validation macro-F1 and QWK | Phase 10 |
| R15 | Overfitting prevention | Row 5 | Dropout, augmentation, L2, overfitting-gap plot (#6, #7) | Sensible regularisation strength (avoid #8's L2 problem) | Phases 9–10 |
| R16 | Organised experiments | Row 5 | With/without comparison design (#6) | Ablation plan, experiment log, fixed seeds | Phase 11 |
| R17 | Accuracy and loss curves | Row 6 (15) | History plots (#2, #7) | Joined curves across both training phases | Phases 10, 12 |
| R18 | Precision, recall, F1 | Row 6 | `classification_report` (#7, #8, #10) | Macro vs weighted averages; sensitivity/specificity for Yes/No | Phase 12 |
| R19 | Confusion matrix | Row 6 | Confusion matrices (#7, #8, #10) | Reading a normalised matrix for ordered classes | Phase 12 |
| R20 | Interpretation and error analysis | Row 6 | Misclassified grid (#6); Grad-CAM (#7) | "How far off" error analysis; Grad-CAM that works with EfficientNet | Phase 12 |
| R21 | Detect DR **and** its stage | Brief task 3 | Softmax multi-class output (#2, #3) | Two-step decision rule (section 0) | Phases 9, 12, 13 |
| R22 | Commented, modular, original code | Row 7 (10) | Commented notebooks, Config class | Shared modules, seeds, pinned versions | All phases |
| R23 | Report ≤ 20 pages with evidence for every step | Row 8 (10) | Lecturer PDFs as style examples | Page budget and figure selection | Phases 14–15 |
| R24 | Working prototype + hosted video URL | Row 8 | Streamlit apps (#11, #12); saving models (#7) | TensorFlow on Apple M2; screen recording and hosting | Phases 13, 16 |
| R25 | Innovation, impact, deployment, ethics, future work | Row 9 (5) | Grad-CAM (#7) | Clinical validation, dataset shift, regulation | Phases 13, 15 |
| R26 | LO2: image acquisition / display / transmission tech | LO map | — | Fundus cameras, smartphone imaging, file formats and compression, tele-ophthalmology | Report §9 |
| R27 | LO3: filtering, segmentation, labelling, feature extraction | LO map | CNN feature maps (slides); U-Net (#9) | CLAHE/unsharp filters, retina segmentation, image hashing | Phases 6–7 |

**Summary:** most of the modelling knowledge is already in your practicals. The main new areas are:

- **Fundus-specific preprocessing:** CLAHE, sharpening, retina cropping.
- **Leakage control:** duplicates, one saved split.
- **Better imbalance handling and metrics:** oversampling, macro-F1, QWK, Yes/No metrics.
- **A few Keras details:** BatchNorm during fine-tuning, the correct input range, Grad-CAM that works with EfficientNet.
- **Report context:** LO2 technologies, ethics.

---

## 2. Phase 4 — Research summary (what the evidence says)

- **APTOS 2019 dataset (Kaggle):**
  - 3,662 labelled training images (≈ 8.6 GB) and 1,928 test images whose labels were never released.
  - Each image was graded 0–4 by a clinician; the images were captured at Aravind Eye Hospital, India.
  - The organisers warn about noise in both images and labels: "Images may contain artifacts, be out of focus, underexposed, or overexposed", and they "were gathered from multiple clinics using a variety of cameras over an extended period of time".
- **Duplicates:** at least one published APTOS solution removed duplicate training images before training. We will check for duplicates ourselves.
- **Patient-level leakage:** in datasets with both eyes per patient (e.g., EyePACS), splitting by image rather than by patient inflates scores, because the two eyes are very similar. APTOS has no patient IDs, so we cannot fully rule this out; we will report it as a limitation.
- **Ben Graham preprocessing (2015 Kaggle DR winner):**
  - Rescale every image to the same eye radius.
  - Subtract the local average colour: `4·img − 4·GaussianBlur(img) + 128`.
  - Clip to 90 % of the radius to remove boundary effects.
  - Augmentation used: ±10 % scaling, 0–360° rotation, small skew. Colour-space augmentation did *not* help.
- **CLAHE evidence:** one study (Hayati et al., 2023) found that CLAHE improved VGG16 (+4 %), InceptionV3 (+5 %) and EfficientNetB4 (+2.8 %) but **hurt** ResNet34 (−11 %). So enhancement must be **tested, not assumed** → Experiment E2.
- **Keras facts:**
  - EfficientNet includes its own rescaling and expects **0–255** pixels. Its native resolutions are B0 = 224, B2 = 260 and B3 = 300 px.
  - Sizes: B0 has 5.3 M parameters, B3 12.3 M, MobileNetV2 3.5 M, ResNet50 25.6 M.
  - The official transfer-learning guide says: keep BatchNorm layers in inference mode when fine-tuning, use a very low learning rate (≈ 1e-5), and re-compile after changing `trainable`.
- **Kaggle Notebooks:**
  - About 30 GPU hours per week (P100, or 2 × T4), roughly 12-hour sessions, and about 20 GB of output space in `/kaggle/working` (figures from a third-party guide — we will check the numbers shown in your Kaggle account).
  - GPU access requires a **verified phone number**.
- **TensorFlow on Mac:** `pip install tensorflow` supports Apple silicon (macOS 12+); there is no official Mac GPU support. That is fine for single-image predictions in the app.

---

## 3. Phase 5 — Strategy (What → Why → How → Alternative)

### 3.1 Problem understanding
- **What:** a 5-class, *ordered* image classification problem (stages 0–4), plus a Yes/No DR decision derived from it.
- **Why:** this matches the brief ("classify DR as well as the stage"). Because the stages are ordered, a 0→4 mistake is worse than a 1→2 mistake.
- **How:** softmax over 5 stages, then the two-step decision rule (section 0). Evaluate with standard metrics plus **QWK** (quadratic weighted kappa). **[Additional knowledge]** QWK scores agreement on ordered classes and penalises big mistakes more than small ones.
- **Alternative:** treating the task as regression (predict a number 0–4, then round). It is popular in Kaggle solutions but harder to explain and to link to precision, recall and F1, so we don't use it.

### 3.2 Dataset selection
- **What:** APTOS 2019 original-resolution training images (3,662 PNGs), attached directly to a Kaggle Notebook.
- **Why:** it is a real clinical dataset with the 0–4 scale. Its size is manageable in 16 days. Using the originals (not the 224 px copy the lecturer used) lets us do our **own** preprocessing at higher resolution — tiny lesions need detail — and makes the work more original.
- **How:** add the competition data to the notebook (accept the competition rules if Kaggle asks) or use a public mirror of the same images. Keep `train.csv` (`id_code`, `diagnosis`) as the label source.
- **Alternatives:**
  - `sovitrath` 224×224 copy: fast, but low-resolution and identical to the lecturer's example.
  - EyePACS 2015: 35k images and patient IDs, but ~80 GB — too heavy for our timeline.
  - IDRiD / Messidor: too small to train on, but possible *external test sets* for future work.

### 3.3 Dataset exploration
- **What:** class counts; image width/height; brightness, contrast and sharpness statistics; sample grid per stage; duplicate check; examples of poor-quality images.
- **Why:** rubric row 1 wants a detailed dataset description, and exploration reveals problems (imbalance, varied cameras, duplicates) that drive later decisions.
- **How:** pandas + Matplotlib in notebook 1.
  - **[Additional knowledge]** *difference hashing (dHash)*, written with OpenCV + NumPy: shrink each image to 9×8 grey pixels, compare neighbouring pixels to get a 64-bit "fingerprint", and treat images whose fingerprints differ by only a few bits as near-duplicates.
  - Image sharpness is measured with the *variance of the Laplacian*.
- **Alternative:** the `imagehash` library. It is easier, but it needs internet to install and hides the logic. Writing our own dHash is short and more original.

### 3.4 Class distribution
- **What:** document the imbalance (e.g., No DR ≈ 49 %, Severe ≈ 5 %) with a bar chart and a table, before and after cleaning.
- **Why:** imbalance explains why accuracy alone misleads — the lecturer notebook scored 53.6 % while predicting only "No DR".
- **How:** charts per split. Handling the imbalance is covered in 3.9.
- **Alternative:** none needed. This is description, not a design choice.

### 3.5 Train / validation / test split
- **What:** one **stratified 70 / 15 / 15** split, made *after* duplicate grouping, with a fixed seed and saved to `splits.csv`. That gives roughly 2,560 / 550 / 550 images.
- **Why:**
  - Stratified keeps the stage proportions equal in every split.
  - Duplicate groups stay together, so the same picture can't sit in both train and test.
  - Saving the split means every experiment uses exactly the same data.
  - The test set is used **once**, at the very end.
- **How:** scikit-learn `train_test_split(..., stratify=...)`, done twice (first split off the test set, then split the rest into train and validation), keeping duplicate groups together. Duplicate groups whose images carry *conflicting* labels will be removed and reported.
- **Alternatives:**
  - 5-fold cross-validation: more reliable, but about 5× the GPU time.
  - The official APTOS test set: impossible to use, because its labels are hidden.

### 3.6 Image preprocessing (standardisation)
- **What:** a deterministic pipeline applied to every image:
  1. Read the image and convert BGR → RGB.
  2. **[Additional knowledge]** *Retina (field-of-view) segmentation:* threshold the dark background, find the retina's bounding box and crop to it.
  3. Pad to a square.
  4. Resize to 512 px (cache size).
  5. Apply a circular mask to clean the corners.
- **Why:**
  - APTOS images come from different cameras, with different sizes and black borders.
  - Cropping makes the retina fill the frame, so small lesions get more pixels.
  - Each step uses only the image itself (no statistics from other images), so there is **no leakage**.
- **How:**
  - One file, `dr_preprocessing.py`, used by the Kaggle notebooks **and** the Streamlit app, so the app sees exactly what the model saw in training.
  - Preprocessed images are cached once as PNGs and saved as a private Kaggle Dataset.
  - Pixels stay 0–255, because EfficientNet does its own scaling.
- **Alternatives:**
  - Preprocess on the fly during training: slower, since OpenCV can't run inside the fast TensorFlow graph.
  - Plain resize without cropping: wastes pixels on black background.

### 3.7 Image enhancement
- **What:** three variants, compared in an experiment:
  - **(A) Basic:** crop + resize only.
  - **(B) CLAHE + sharpening** **[Additional knowledge]**:
    - CLAHE (Contrast Limited Adaptive Histogram Equalisation) boosts contrast in small tiles, so faint lesions stand out without over-brightening the image. It is applied to the lightness channel of the LAB colour space, which keeps the colours natural.
    - Unsharp masking adds back fine detail: `sharp = img + k·(img − blur(img))`. This covers the brief's "edge enhancement".
  - **(C) Ben Graham:** subtracting the blurred image evens out lighting across different cameras.
- **Why:**
  - The brief explicitly asks for contrast adjustment and edge enhancement.
  - Research shows enhancement can help or hurt depending on the model, so we measure its effect instead of assuming it.
- **How:**
  - OpenCV: `cv2.createCLAHE`, `cv2.GaussianBlur`, `cv2.addWeighted`.
  - Before/after figure for each step.
  - Simple quality numbers (contrast, sharpness) before vs after.
  - Experiment **E2** decides which variant the final model uses.
- **Alternatives:**
  - Green-channel-only processing: vessels and lesions are clearest there, but pretrained networks expect 3 colour channels.
  - Global histogram equalisation: often over-boosts bright regions.
  - Median/bilateral denoising: tried visually; only kept if it clearly helps.

### 3.8 Data augmentation
- **What:** applied to the **training set only**, on the fly:
  - flips (horizontal and vertical);
  - rotations of any angle (0–360°);
  - zoom ±10 %;
  - small shifts;
  - brightness/contrast ±15 %.
- **Why:**
  - Fundus photos have no natural "up", and left and right eyes are mirror images, so flips and rotations are realistic.
  - Brightness and contrast changes imitate different cameras.
  - We avoid **hue/colour shifts** (lesion colour matters, and Ben Graham found colour augmentation didn't help) and **cutout** (it could erase the only lesion).
- **How:**
  - **[Additional knowledge]** A `tf.data` pipeline with Keras preprocessing layers (`RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomTranslation`, `RandomBrightness`, `RandomContrast`). This replaces `ImageDataGenerator`, which is now deprecated.
  - These layers are active only during training.
  - Validation and test images get **no** augmentation.
  - An augmentation-examples figure goes in the report.
- **Alternative:** `ImageDataGenerator`, as in your practicals. It is familiar but deprecated and slower. Optional experiment E6 compares augmentation off / light / strong.

### 3.9 Class imbalance handling
- **What:** compare three settings on the same model:
  - no balancing (baseline);
  - **class weights** (existing knowledge);
  - **oversampling** of rare stages inside the training set **[Additional knowledge]** — rare-stage images are shown more often, and augmentation makes each repeat look different.
- **Why:** rare stages (Severe, Proliferative) matter most clinically but are easy for a model to ignore. The rubric asks for imbalance to be "handled appropriately".
- **How:**
  - Class weights come from the **training** split only (`compute_class_weight('balanced')`).
  - Oversampling uses `tf.data.Dataset.sample_from_datasets` with per-class streams.
  - Validation and test keep the real distribution, so the results stay honest.
  - The two methods are never combined, to avoid double-counting.
  - Judged by **validation macro-F1** and per-class recall (experiment **E3**).
- **Alternatives:**
  - **[Additional knowledge]** Focal loss: focuses training on hard examples. Optional extra run.
  - Undersampling "No DR": throws data away.
  - Synthetic images (GANs): far too complex for this timeline.

### 3.10 CNN architecture selection
- **What:** **EfficientNetB0** as the main experimental backbone, **EfficientNetB3** as the larger final candidate, and **MobileNetV2** as a familiar baseline.
- **Why:**
  - EfficientNet scales depth, width and input resolution together **[Additional knowledge: compound scaling]**, giving strong accuracy for few parameters. B0 has 5.3 M parameters at 224 px; B3 has 12.3 M at 300 px.
  - Higher resolution matters for tiny lesions, which is why B3 is worth testing.
  - Starting with the small B0 keeps experiments fast within the GPU quota.
  - MobileNetV2 is already familiar from your practicals, so the comparison shows your knowledge.
  - We are not choosing EfficientNet because it is popular; experiment **E5** confirms the choice with our own numbers.
- **How:** `keras.applications.EfficientNetB0/B3(include_top=False, weights='imagenet')` plus a small head: GlobalAveragePooling → Dropout(0.3) → Dense(5, softmax).
- **Alternatives:**
  - ResNet50: 25.6 M parameters, heavier for similar accuracy.
  - DenseNet121: popular in medical imaging.
  - VGG16: 138 M parameters, too heavy.
  - Newer models (ConvNeXt, Vision Transformers): heavier and harder to justify for 3.6k images.

### 3.11 Transfer learning
- **What:** two stages.
  1. **Feature extraction:** freeze the pretrained base and train only the new head.
  2. **Fine-tuning:** unfreeze the top part of the base (about the last 20–30 % of layers) and keep training with a very small learning rate.
- **Why:**
  - ImageNet features (edges, textures, blobs) transfer well, but retinal lesions differ from everyday photos, so partial fine-tuning adapts the top layers.
  - Freezing first stops the random new head from damaging the pretrained weights.
- **How:**
  - Pixels go in as 0–255.
  - **[Additional knowledge]** BatchNorm layers stay frozen / in inference mode during fine-tuning (official Keras advice).
  - Re-compile after changing `trainable`.
  - Fine-tuning learning rate ≈ 1e-5 to 1e-4.
  - Build the model so its inner layers can be reached, so Grad-CAM works (the lecturer notebook failed here).
- **Alternatives:**
  - Train all layers from the start: risky with 2.5k images.
  - Frozen only: faster but usually weaker. Experiment **E4** measures the difference.

### 3.12 Training strategy
- **What:**
  - Adam optimiser; sparse categorical cross-entropy.
  - Up to ~15 epochs for the head and ~20 for fine-tuning, with early stopping.
  - Callbacks: **ModelCheckpoint** (best validation macro-F1), **EarlyStopping** (patience 5, restores the best weights), **ReduceLROnPlateau** (halves the LR after 2–3 flat epochs), **CSVLogger**.
  - **[Additional knowledge]** A small custom callback that computes validation macro-F1 and QWK after every epoch.
- **Why:**
  - These are exactly the items rubric row 5 lists.
  - Choosing the model by macro-F1 (not accuracy) prevents the "predict everything as No DR" trap.
- **How:**
  - Fixed seeds (42).
  - Cached preprocessed images → `tf.data` (shuffle, augment, batch, prefetch).
  - Every run appends a row to `experiment_log.csv`.
  - The best model is saved as `.keras`.
  - We test loading a saved model on your Mac **early** (Phase 10) to catch version problems.
- **Alternatives:**
  - Cosine learning-rate decay (you used it in the PyTorch practical).
  - AdamW (weight decay built in).
  - Mixed precision for speed on T4 GPUs (optional).

### 3.13 Hyperparameter selection

| Hyperparameter | Starting value | Tuned? |
|---|---|---|
| Input size | 224 (B0 / MobileNetV2), 300 (B3) | E5 |
| Batch size | 32 (B0), 16 (B3) | Fixed by GPU memory |
| Head learning rate | 1e-3 | Small check (1e-3 vs 3e-4) |
| Fine-tune learning rate | 1e-5 | Small check (1e-5 vs 1e-4) |
| Unfrozen share of base | ~25 % of layers | E4 |
| Dropout | 0.3 | Fixed unless overfitting is seen |
| Epochs | Early stopping (patience 5) | — |
| Enhancement variant | A / B / C | E2 |
| Balancing method | none / weights / oversampling | E3 |

- **Why:** a few targeted checks on the **validation** set are affordable and explainable.
- **How:** change one thing at a time, keeping everything else fixed.
- **Alternative:** Keras Tuner (automatic search). More GPU time and a "black box", so it's optional only.

### 3.14 Overfitting prevention
- **What:** augmentation, dropout, partial fine-tuning, early stopping with best-weight restore, a moderate model size, and watching the train–validation gap.
- **Why:** 2.5k training images are few for a CNN; the rubric asks for overfitting control explicitly.
- **How:** plot train vs validation accuracy and loss for both stages, and add the **overfitting-gap plot** from your augmentation practical. We avoid very strong L2 (the lecturer notebook's loss was dominated by it).
- **Alternatives:**
  - Label smoothing.
  - `drop_connect_rate` (EfficientNet's built-in stochastic depth).
  - Weight decay.

### 3.15 Model evaluation (test set, once)
- **What:**
  - Accuracy; per-stage precision, recall and F1; macro and weighted averages; classification report.
  - Confusion matrix (counts and row-normalised); QWK.
  - Yes/No DR metrics: sensitivity, specificity, precision, F1, ROC-AUC.
  - Training/validation curves for both stages.
- **Why:**
  - These are all listed in rubric row 6.
  - QWK respects stage order.
  - Sensitivity matters most in screening: missing a sick eye is worse than a false alarm.
- **How:** scikit-learn metrics in notebook 4, which loads the saved best model and the saved split. Each metric gets a plain-English explanation in the report.
- **Alternatives / extras:**
  - Bootstrap confidence intervals **[Additional knowledge]**: re-sample the test set many times to show how uncertain the numbers are, since small classes make them noisy.
  - A "referable DR" view (stage ≥ 2).

### 3.16 Error analysis
- **What:**
  - Which stages get confused (we expect neighbouring stages, e.g., Mild↔Moderate).
  - How far off the mistakes are (off by 1 vs 2+ stages).
  - Recall vs class size (the effect of imbalance).
  - Grids of wrong predictions with their probabilities.
  - **Grad-CAM** heatmaps for correct and wrong cases.
  - Whether blurry or dark images fail more often.
- **Why:** the rubric asks for "insightful interpretation … and error analysis". Grad-CAM shows whether the model looks at lesions or at artefacts like borders and reflections.
- **How:** reuse ideas from your practicals (misclassified grid, Grad-CAM) with our own code, plus the sharpness/brightness numbers from Phase 6.
- **Alternative:** saliency maps or LIME. Grad-CAM is standard and already familiar.

### 3.17 Prototype development
- **What:** a **Streamlit** web app running on your Mac:
  1. Upload a fundus image.
  2. See the preprocessing steps.
  3. See **"DR present? Yes/No (with %)"** and **"Stage: …"**, plus a bar chart of all 5 stage probabilities.
  4. See the Grad-CAM heatmap.
  5. Get an image-quality warning (blurry/dark) — optional.
  6. A clear disclaimer: "educational prototype, not a medical device".
- **Why:** you have already built Streamlit apps (#11, #12), and showing preprocessing and Grad-CAM live makes the video stronger and adds innovation marks.
- **How:** `app.py` imports the same `dr_preprocessing.py` and loads `best_model.keras`. The Python environment on the Mac uses the **same TensorFlow version** as Kaggle.
- **Alternatives:**
  - Gradio: similar and simpler, but new to you.
  - Hosting the app online (Streamlit Community Cloud / Hugging Face Spaces): optional; the brief only needs a hosted *video*.

### 3.18 Report evidence (≤ 20 pages)

| Report section | ~Pages | Key evidence |
|---|---|---|
| Title, abstract, contents | 1 | — |
| 1. Problem & objectives | 1.5 | Stage table; system overview diagram |
| 2. Dataset | 2.5 | Class chart, sample grid, image-size plot, duplicate examples, split table, limitations |
| 3. Preprocessing | 2 | Step-by-step before/after figure; quality numbers |
| 4. Augmentation & balancing | 1.5 | Augmentation grid; balance before/after chart |
| 5. Model & transfer learning | 2.5 | Architecture diagram; hyperparameter table; key code snippet |
| 6. Training & experiments | 2.5 | Experiment table (E1–E5); curves; overfitting-gap plot |
| 7. Evaluation & error analysis | 3 | Confusion matrices, classification report, Yes/No metrics, Grad-CAM panel, error grid |
| 8. Prototype & video | 1.5 | 2–3 app screenshots; **video URL** |
| 9. Discussion | 1.5 | Impact, deployment, ethics, LO2 technologies, limitations, future work |
| References | 0.5 | — |

Every figure gets a caption and a sentence explaining why it matters. There are no decorative screenshots.

**How the report gets written.** Mohamed writes it. Claude's part is to supply the raw material for each section — the figures, the tables, the measured numbers, explanations of what a result means, and suggested structure — and to say plainly where each number came from so it can be checked against the notebook that produced it. Claude does not produce a finished report to be submitted as-is, and never writes a figure, metric or claim that has not come out of a run Mohamed can reproduce. Anything uncertain is flagged as uncertain rather than smoothed over.

### 3.19 Video demonstration
- **What:** a 10–12 minute recording (the lecturer's limit is 20). Outline:
  1. Problem (1 min)
  2. Dataset (1.5)
  3. Preprocessing (1.5)
  4. Augmentation and balancing (1)
  5. Model and training (2)
  6. Results (2)
  7. **Live app demo** (2.5)
  8. Limitations and ethics (1)
- **Why:** the lecturer said the report and video are the main assessed items, so the video must show the prototype working.
- **How:**
  - Record with macOS screen recording (Shift-Command-5) or OBS.
  - Upload to YouTube as **Unlisted** (or Google Drive with "anyone with the link").
  - Test the link in a private browser window, then paste it into the report.
- **Alternative:** a narrated slide deck with embedded clips — still needs the live demo.

---

## 4. System workflow

```
[Kaggle Notebook 1 — Explore & split]
  APTOS train.csv + 3,662 PNGs
    → exploration figures (classes, sizes, quality)
    → dHash duplicate check → remove conflicting-label duplicates
    → stratified 70/15/15 split (seed 42) → splits.csv
        ↓
[Kaggle Notebook 2 — Preprocess]
  dr_preprocessing.py (shared)
    → retina crop → pad → resize 512 → circular mask
    → variants: A basic | B CLAHE + unsharp | C Ben Graham
    → before/after figures → cached PNGs saved as a private Kaggle Dataset
        ↓
[Kaggle Notebook 3 — Train & experiments]
  tf.data (train: shuffle + augment + [weights | oversampling]; val: no augmentation)
    → EfficientNet/MobileNetV2 + head
    → stage 1: frozen base → stage 2: fine-tune top layers (BatchNorm frozen)
    → callbacks + macro-F1/QWK logging → experiment_log.csv → best_model.keras
        ↓
[Kaggle Notebook 4 — Final evaluation (test set used once)]
  metrics, confusion matrices, Yes/No metrics, curves, Grad-CAM, error analysis → figures
        ↓
[Mac M2 — Streamlit prototype]
  app.py + dr_preprocessing.py + gradcam.py + best_model.keras
    → upload → preprocess → DR Yes/No + stage + probabilities → Grad-CAM → disclaimer
        ↓
[Report PDF ≤ 20 pages + hosted video URL]
```

### Leakage checklist
- [ ] Duplicates grouped before splitting; conflicting-label groups removed and reported.
- [ ] One saved split used by every notebook.
- [ ] Preprocessing uses only each image's own pixels (no dataset statistics).
- [ ] Augmentation and oversampling only in the training pipeline.
- [ ] Class weights computed from the training split only.
- [ ] Model choice, thresholds and hyperparameters decided on validation only.
- [ ] Test set evaluated once, in notebook 4.
- [ ] Patient-level limitation stated in the report.

---

## 5. Experiment plan (Phase 11)

All runs use the same split, seed and callbacks and are compared on the **validation set**.

| ID | Objective | What changes | Why |
|---|---|---|---|
| E1 | Baseline | EfficientNetB0 @224, variant A (basic), frozen base, light augmentation, no balancing | Reference point |
| E2 | Does enhancement help? | Variant B (CLAHE + unsharp) and variant C (Ben Graham) vs A | Brief asks for enhancement; evidence is mixed |
| E3 | Does balancing help rare stages? | Best of E2 + class weights vs oversampling | Imbalance handling (rubric row 3) |
| E4 | Frozen vs fine-tuned | Best of E3 + fine-tune top ~25 % of layers | Transfer-learning depth (rubric row 4) |
| E5 | Bigger model / higher resolution | Best settings on EfficientNetB3 @300 and MobileNetV2 @224 | Justifies the architecture with evidence |
| E6 (optional) | Augmentation strength | None vs light vs strong | Only if time allows |

**Recorded for every run:**
- objective, change and reason;
- validation accuracy, macro-F1, QWK and per-stage recall;
- best epoch and training time;
- curves;
- a short interpretation.

The winner of E5 becomes the final model and is evaluated once on the test set. **Rough GPU budget:** 6–10 hours in total, well inside the weekly quota.

---

## 6. File locations

**On Kaggle (your account):** four notebooks — `dr-01-explore-split`, `dr-02-preprocess`, `dr-03-train-experiments`, `dr-04-evaluate` — plus one private dataset, `dr-aptos-preprocessed`, holding the cached images and `splits.csv`.

**On your Mac** — suggested location, created in Phase 6, inside the already-connected module folder:

```
Computer vision/
└── CW1_DR_Stage_Detection/
    ├── kaggle_notebooks/        ← the .ipynb files we write (uploaded to Kaggle)
    ├── app/
    │   ├── app.py               ← Streamlit prototype
    │   ├── dr_preprocessing.py  ← same file used on Kaggle
    │   ├── gradcam.py
    │   └── requirements.txt
    ├── models/best_model.keras  ← downloaded from Kaggle
    ├── results/                 ← experiment_log.csv, metrics
    ├── report_figures/          ← figures for the report
    └── report/                  ← report drafts and final PDF
```

---

## 7. Timeline (16 days)

| Dates | Phase(s) | Output |
|---|---|---|
| 17 Sep | 3–5 | This plan ✔; Kaggle account checks |
| 18–19 Sep | 6 | Notebook 1: exploration, duplicates, `splits.csv` |
| 20–21 Sep | 7 | Notebook 2: preprocessing + cached dataset |
| 21–22 Sep | 8–9 | Pipeline, augmentation, balancing, model builder |
| 22–23 Sep | 10 | E1 baseline; test model loading on the Mac |
| 23–26 Sep | 11 | E2–E5 |
| 26–27 Sep | 12 | Final test evaluation, Grad-CAM, error analysis |
| 27–29 Sep | 13–14 | Streamlit app; final figures and screenshots |
| 29 Sep – 2 Oct | 15–16 | Report; video recorded and uploaded |
| 3 Oct | — | Final checks and Turnitin submission |

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| GPU quota or session timeouts | Use B0 for experiments; save checkpoints; use Kaggle "Save Version" to run in the background |
| Rare stages have few test images (e.g., ~29 Severe) | Report per-class counts; optional bootstrap confidence intervals; discuss uncertainty |
| Label noise / camera variety | Discuss as a limitation; inspect in error analysis |
| TensorFlow version mismatch between Kaggle and Mac | Pin the same version; test loading early |
| Time pressure | The optional items (E6, bootstrap, online hosting) are dropped first |

---

## References (for the report later)
- Kaggle — APTOS 2019 Blindness Detection (competition data), https://www.kaggle.com/competitions/aptos2019-blindness-detection/data; size and file list via Academic Torrents mirror, https://academictorrents.com/details/d8653db45e7f111dc2c1b595bdac7ccf695efcfd
- Wilkinson C.P. et al. (2003) "Proposed international clinical diabetic retinopathy and diabetic macular edema disease severity scales", *Ophthalmology* 110(9), https://pubmed.ncbi.nlm.nih.gov/13129861/
- Graham B. (2015) "Kaggle Diabetic Retinopathy Detection competition report", https://kaggle-forum-message-attachments.storage.googleapis.com/88655/2795/competitionreport.pdf
- Hayati M. et al. (2023) "Impact of CLAHE-based image enhancement for diabetic retinopathy classification through deep learning", *Procedia Computer Science* 216, 57–66, https://www.sciencedirect.com/science/article/pii/S1877050922021895
- Zaylaa A.J. & Kourtian S. (2025) "From Pixels to Diagnosis…", *Applied Sciences* 15(5), 2684, https://www.mdpi.com/2076-3417/15/5/2684
- Keras — Transfer learning & fine-tuning guide, https://keras.io/guides/transfer_learning/ ; EfficientNet fine-tuning example (input resolutions), https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/ ; Keras Applications table, https://keras.io/api/applications/
- TensorFlow — pip installation (macOS / Apple silicon), https://www.tensorflow.org/install/pip
- abhuse — APTOS 2019 solution (duplicate removal), https://github.com/abhuse/aptos-retinopathy-detection
- NECO — Diabetic retinopathy staging summary, https://www.neco.edu/diabetic-retinopathy-staging-exam-imaging/
