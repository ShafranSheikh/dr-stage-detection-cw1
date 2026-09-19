# Phase 1 — Practical Background & Assignment Analysis

**Module:** Computer Vision — CW1 "Diabetic Retinopathy Stage Detection" (NIBM, BSCCOMP24.2P)
**Prepared:** 17 September 2026 · **Working deadline:** 3 October 2026
**Sources inspected:** the local `Computer vision` folder (16 files, no subfolders) and the CW descriptor in this project.

---

## 0. What was inspected

| Type | Files |
|---|---|
| Jupyter notebooks | `01_Hand_written_Character_Recognition`, `02_imdb_moview_review_classification_tf_hub`, `Deep_Learning_Image_Classification` (+ an identical "(1)" copy), `Deep_Neural_Network`, `data_augmentation_comparison`, `Tumor_prediction_with_image_Masks`, `siamese_network_tutorial`, `Face_Recognition_Based_Attendance`, `Defense_Multi_Agent_LLM_fast` |
| Notebooks saved in other formats | `transfer_learning_tf.ipynb.txt` (a full notebook), `Colab_Eye_Disease_Prediction.ipynb.html` (a notebook downloaded from the NIBM LMS) |
| Lecture material | `Convolutional Neural Networks- CNN (1).pptx` (28 slides, Dr. Kaneeka Vidanage) |
| Lecturer guides (PDF) | `Diabetic Retinopathy Detection Using Deep Learning.pdf` (7 pp.), `Brain_MRI_Segmentation_Kaggle_UNet_Dice_Report.pdf` (10 pp.) |
| Other | `requirements.txt` (tensorflow, numpy, pandas) |

---

## A. Practical overview

| # | File | Topic | Concepts & techniques shown | Stack | Status of saved run |
|---|---|---|---|---|---|
| 1 | CNN slides | CNN fundamentals | Image = RGB pixel matrices; why fully-connected nets don't scale; local connectivity; convolution (filter × patch → feature map); ReLU; max pooling (2×2, stride 2); stacking layers; fully-connected classification (X vs O example) | — | Lecture |
| 2 | `01_Hand_written_Character_Recognition` | MNIST digits: MLP vs CNN | Normalisation (÷255), one-hot labels, reshape to (N, H, W, 1), Conv2D + MaxPooling + Dropout, Adam, validation split, accuracy/loss curves, single-image prediction | TF/Keras | Ran — MLP 97.5 %, CNN 99.1 % test accuracy |
| 3 | `Deep_Learning_Image_Classification` | Fashion-MNIST MLP | Flatten/Dense, softmax, sparse categorical cross-entropy, `argmax` → class name | TF/Keras | Ran — 87.9 % test accuracy |
| 4 | `Deep_Neural_Network` | Pima diabetes (tabular data) | Dense layers, sigmoid, binary cross-entropy, `train_test_split` | Keras, pandas, scikit-learn | Ran — ~71 %; not image data |
| 5 | `02_imdb…tf_hub` | Movie-review sentiment | Transfer learning with a pretrained (trainable) TF-Hub embedding layer | TF, TF-Hub | Ran — text, not images; useful only for the transfer-learning idea |
| 6 | `data_augmentation_comparison` | CIFAR-10: augmentation vs none | Flips, random crop, colour jitter, rotation; test data never augmented; BatchNorm CNN; AdamW + cosine LR schedule; seeds; overfitting-gap plot; per-class accuracy; misclassified-image grid | PyTorch, torchvision | Training ran (augmented model ≈ 89.7 %); comparison plots not saved |
| 7 | `transfer_learning_tf` | Kaggle Flowers — full transfer-learning pipeline | Kaggle API; 70/15/15 split per class; class-distribution chart; `ImageDataGenerator` augmentation on train only; MobileNetV2 `preprocess_input`; frozen base → fine-tune top 30 layers at LR 1e-5; GAP + BatchNorm + Dense + Dropout + L2 head; EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger; confusion matrix; classification report; per-class F1; Grad-CAM; saving `.keras`/`.h5` | TF/Keras, scikit-learn, seaborn, OpenCV | Partly ran — phase 1 best val accuracy 91.2 %; phase 2 crashed (TensorBoard missing); evaluation/Grad-CAM cells not run |
| 8 | `Colab_Eye_Disease_Prediction` + DR guide PDF | **Diabetic retinopathy grading (our topic)** | Kaggle `sovitrath/diabetic-retinopathy-224x224-2019-data` (APTOS-2019, 3,662 images); black-border crop and Ben Graham filter (defined); stratified 70/10/20 split; augmentation; class weights; EfficientNetB3 two-phase training; focal loss (defined, unused); quadratic weighted kappa (QWK); confusion matrix; Grad-CAM; single-image inference | TF/Keras, OpenCV, scikit-learn | Ran — **model failed** (see lessons below) |
| 9 | `Tumor_prediction_with_image_Masks` + Brain MRI PDF | Brain MRI tumour segmentation | Kaggle API; image/mask pairing; grayscale read, resize, normalise; U-Net with skip connections; Dice loss; why accuracy misleads on imbalanced pixels | TF/Keras, OpenCV | Ran — loss stayed flat, so the model did not learn |
| 10 | `siamese_network_tutorial` | Face verification (AT&T faces) | Pair generation, shared-weight CNN encoder, contrastive loss, stratified train/val/test, callbacks, classification report, confusion matrix, ROC-AUC, t-SNE, one-shot demo, model saving | TF/Keras, scikit-learn, seaborn | Not executed (no outputs) |
| 11 | `Face_Recognition_Based_Attendance` | Face-recognition attendance app | OpenCV DNN (SSD) face detector, ArcFace ONNX embeddings, cosine-distance threshold, **Streamlit web app** (upload/camera input, sliders, caching), CSV logging | OpenCV, ONNX Runtime, Streamlit, pandas | Ran (local Windows/conda) |
| 12 | `Defense_Multi_Agent_LLM_fast` | LLM multi-agent workflow | LangGraph, Ollama, Streamlit | LangChain | Not computer vision — out of scope |

### Lessons from the saved runs (important)

**The lecturer's DR notebook (#8) is the closest example, but its saved run failed.** Understanding why is valuable, and our code must be original anyway (the lecturer knows this notebook).

- **A class was silently lost.** The folder is named `Proliferate_DR`, but the label map expected `Proliferative_DR`, so all 295 proliferative images were dropped. The model trained on 4 classes while having 5 outputs.
- **Wrong input range.** Images were rescaled to 0–1, but Keras EfficientNet already contains its own rescaling layer and expects pixels in the 0–255 range. The network effectively saw almost-black images.
- **The loss was mostly the regulariser.** L2 = 0.01 on two large Dense layers accounts for most of the ~11–13 loss value (about 10–11, estimated from the layer sizes), so training mainly shrank weights instead of learning to classify.
- **Half the epochs barely trained.** Every second epoch finished in 7–10 s with batch-sized accuracy values, which suggests the data generator ran out part-way (a `steps_per_epoch` mismatch). `validation_steps` also skipped 17 validation images.
- **Result:** every test image was predicted as "No DR" → test accuracy 53.6 % (exactly the share of No-DR images), QWK 0.00, macro-F1 0.17.
- **Grad-CAM failed** ("No convolutional layer found") because the conv layers sit inside the nested EfficientNet sub-model.
- **Ben Graham preprocessing was defined but never used** in training.

**Takeaway:** accuracy alone would have looked "acceptable". This is exactly why the rubric demands precision, recall, F1 and a confusion matrix.

**Other method/leakage lessons**

- #6 evaluates on the test set every epoch, so the test set acts as a validation set.
- #10 chooses its decision threshold on the test set. It also builds image pairs before splitting, so the same faces appear in train and test.
- #9 splits MRI slices randomly, so slices from the same patient end up in both train and test.
- #9's validation loss and accuracy (~98.8 %) stayed identical for all 20 epochs — a sign it predicted "no tumour" everywhere, yet accuracy still looked high.

**Security note:** the saved output of #9 contains a Kaggle API key that isn't yours. Don't reuse it. Use your own `kaggle.json` and keep it out of submitted code and the report.

---

## B. Existing knowledge (what the practicals show you have met)

- **Image fundamentals:** images as arrays (0–255), channels, shapes (N, H, W, C), grayscale vs RGB, OpenCV BGR↔RGB conversion.
- **Basic preprocessing:** resizing (OpenCV/PIL), normalisation (÷255 and model-specific scaling such as MobileNetV2 `preprocess_input` or ArcFace's (x−127.5)/128), adding a channel dimension, cropping black borders (seen in #8).
- **Neural-network basics:** Dense/MLP, ReLU/softmax/sigmoid, one-hot vs sparse labels, categorical/sparse/binary cross-entropy, Adam, batch size and epochs.
- **CNN building blocks:** Conv2D, filters and feature maps, pooling, flatten, fully-connected layers, Dropout, BatchNorm, Global Average Pooling — in Keras and PyTorch.
- **Data augmentation:** flips, rotation, shifts, zoom, shear, brightness, random crop, colour jitter; train-only augmentation; before/after visualisation; measuring the overfitting gap.
- **Transfer learning:** ImageNet MobileNetV2 as a frozen feature extractor, then fine-tuning the top layers with a very small learning rate; EfficientNetB3 (in #8); pretrained models used only for inference (ArcFace, SSD); TF-Hub embeddings.
- **Training control:** validation split, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger, L2 regularisation, AdamW with a cosine schedule, random seeds.
- **Data handling:** Kaggle API (`kaggle.json`, `chmod 600`), folder-per-class datasets, DataFrames of file paths, `flow_from_directory`/`flow_from_dataframe`, stratified `train_test_split`, three-way splits.
- **Class imbalance:** `compute_class_weight('balanced')` (#8); a focal-loss definition (#8, unused).
- **Evaluation:** accuracy/loss curves, confusion matrix, classification report (precision/recall/F1), per-class accuracy and F1 charts, ROC-AUC, QWK (#8), prediction grids with confidence, misclassified examples.
- **Explainability:** Grad-CAM (TF GradientTape), t-SNE.
- **Other CV tasks:** segmentation (U-Net, Dice), metric learning (Siamese, contrastive loss), face detection and recognition.
- **Deployment:** saving/reloading models, single-image inference functions, Streamlit apps, ONNX Runtime.
- **Tools:** TensorFlow/Keras (6 of the 7 image-model notebooks), PyTorch (1), OpenCV, PIL, scikit-learn, pandas, Matplotlib/Seaborn, Google Colab GPU, local Jupyter/Anaconda.
- **Expected style:** Colab/Jupyter notebooks with a `Config` class, step-by-step commented cells, plain-language explanations, and evaluation beyond accuracy.

**Gap noticed:** the folder has no classical image-processing practicals (histogram equalisation, sharpening/edge filters, thresholding, morphology). These matter because the brief asks for "contrast adjustment, edge enhancement".

---

## C. Techniques from the practicals that may be relevant

Final selection depends on suitability and on experiment results — not every technique will be used.

| Technique | Seen in | Possible role in CW1 |
|---|---|---|
| Kaggle API download | #7, #8, #9, #10 | Dataset acquisition (required) |
| DataFrame of paths + stratified split | #7, #8, #10 | Reproducible train/val/test split |
| Black-border cropping, resizing, colour conversion | #8, #11 | Core preprocessing |
| Ben Graham (subtract Gaussian-blurred image) | #8 (defined only) | Illumination/contrast normalisation candidate |
| Backbone-specific input scaling | #7 | Must match the chosen network |
| Train-only augmentation (flips, rotation, zoom, brightness) | #6, #7, #8 | Augmentation (required) |
| Class weights / focal loss | #8 | Balancing candidates |
| Two-phase transfer learning (freeze → fine-tune) | #7, #8 | Core model strategy |
| Callbacks (EarlyStopping, checkpoint, LR scheduling, CSV log) | #7, #8, #10 | Training strategy (rubric row 5) |
| Confusion matrix, classification report, per-class F1, QWK | #7, #8, #10 | Evaluation (rubric row 6) |
| Grad-CAM | #7, #8 | Explainability, error analysis, innovation |
| Overfitting-gap plot, misclassified grid | #6 | Error analysis |
| Model saving + inference function | #7, #8, #10 | Prototype back-end |
| Streamlit app | #11, #12 | Prototype front-end |
| Not needed here | #4, #5, #9, #10, #12 | U-Net/Dice, Siamese, face detection, text models, LLM agents — background only |

---

## D. Additional knowledge (not covered in the practicals)

Each item will be explained in detail before we use it.

1. **CLAHE (Contrast Limited Adaptive Histogram Equalisation):** boosts contrast in small tiles, so tiny lesions become visible without over-brightening the whole image. Covers "contrast adjustment".
2. **Edge enhancement / unsharp masking:** sharpens by adding back the detail (image − blurred image); makes vessel edges and small lesions crisper. Covers "edge enhancement". Classical filters (Sobel, Laplacian) also support LO3.
3. **Retina (field-of-view) segmentation:** threshold + largest contour to find the circular retina, crop to it and mask the black background. Links to LO3 "segmentation".
4. **Duplicate detection (perceptual hashing):** APTOS-2019 is known to contain duplicate images (some Kaggle solutions removed them). A duplicate that lands in both train and test inflates scores.
5. **Patient-level splitting and dataset limits:** APTOS has no patient IDs, so patient separation can't be guaranteed — a limitation to report. EyePACS has left/right-eye IDs per patient.
6. **Ordinal grading and QWK:** DR stages are ordered, so predicting stage 4 for a stage-0 eye is worse than predicting stage 1. QWK measures this. It appeared in #8 but was never computed successfully.
7. **Imbalance strategies beyond class weights:** oversampling rare stages inside the training set only, focal loss, and macro-averaged metrics.
8. **Backbone-specific input handling and BatchNorm during fine-tuning:** EfficientNet expects 0–255 input, MobileNetV2 expects −1 to 1; keep BatchNorm layers in inference mode while fine-tuning.
9. **`tf.data` pipelines + Keras preprocessing layers:** `ImageDataGenerator` (used in #7 and #8) is marked deprecated in current Keras; this is the modern replacement — faster and easier to control.
10. **Binary screening metrics:** sensitivity/specificity (and ROC-AUC) for "DR vs No DR".
11. **Medical-AI context (LO2 and ethics):** fundus cameras and smartphone imaging (acquisition), display and compression, telemedicine transmission, dataset bias, and "not a diagnostic device" framing.

---

## 2. Assignment descriptor analysis

**Basics:** individual work, 100 marks, covers LO1–LO4, submitted via Turnitin on the VLE. The hand-in and hand-out dates on the form look swapped (hand-in "2026-9-03", hand-out "2026-10-03"). The working deadline is **3 October 2026**.

**Tasks in the brief**

1. Organise a Kaggle dataset using basic preprocessing (e.g., contrast adjustment, edge enhancement).
2. Use data augmentation to increase diversity.
3. Use a suitable CNN with transfer learning to classify DR **as well as its stage**.
4. Evaluate with accuracy and loss curves plus precision, recall and F1.

**Deliverables:** commented codebase; **one PDF report of at most 20 pages** with graphical evidence for every step; a hosted video URL of the prototype running.

**Requirements that are easy to miss**

- "DR as well as the stage" → the output should say *whether DR is present* **and** *which stage (0–4)*.
- "Graphical evidence … on all steps" → every pipeline step needs a figure with an explanation.
- LO2 (hardware/software for image acquisition, display and transmission) has no rubric row but is mapped to this CW → add a short section.
- Your lecturer said there is no viva, the video can be up to 20 minutes, the report and video are the main assessed items, and unique implementations earn extra credit. The rubric still gives 10 marks to code quality, so key code should appear cleanly in the report and video, and the code should stay tidy.

**Rubric: where the marks are**

| Criterion | Marks | What "Excellent" requires |
|---|---|---|
| 1. Problem & dataset justification | 10 | DR and its medical significance; relevant Kaggle dataset; classes, distribution, splits, ethics, limitations |
| 2. Preprocessing | 10 | Justified, reproducible pipeline (contrast, resizing, normalisation, noise removal, edge enhancement) that visibly improves images, with no leakage |
| 3. Augmentation & balancing | 10 | Justified augmentation (rotation, flips, zoom, brightness); class imbalance handled |
| 4. CNN & transfer learning | 20 | Suitable architecture (ResNet / EfficientNet / VGG / MobileNet), correct and optimised transfer learning, justified design, hyperparameter tuning |
| 5. Training & experimental design | 10 | Validation, callbacks, early stopping, LR scheduling, overfitting prevention, organised experiments |
| 6. Evaluation & analysis | 15 | Accuracy, precision, recall, F1, confusion matrix, curves, insightful interpretation, error analysis |
| 7. Code quality | 10 | Modular, fully commented, original, reproducible |
| 8. Report & presentation | 10 | Professional, ≤ 20 pages, strong visuals, video link included and explained |
| 9. Innovation & impact | 5 | Originality, healthcare impact, deployment feasibility, limitations, ethics, future work |

Roughly **45 %** of the marks depend on the modelling work (rows 4–6), **30 %** on the data work (rows 1–3) and **25 %** on communication and originality (rows 7–9).

---

## 3. Proposed approach (high level — details in Phases 3–5)

**Pipeline**

Kaggle dataset → exploration (class counts, image sizes, duplicates) → remove duplicates → one stratified train/val/test split (fixed seed, saved to CSV) → preprocessing (retina crop → resize → CLAHE → edge enhancement / Ben Graham) → train-only augmentation + class balancing → pretrained CNN (train head with frozen base → fine-tune top layers) → model selection on validation macro-F1/QWK → **single** test-set evaluation → error analysis + Grad-CAM → Streamlit prototype → report + video.

**Key recommendations** (each will get a full What → Why → How → Alternative in Phase 5)

| Decision | Recommendation | Why | Alternative |
|---|---|---|---|
| Framework | TensorFlow/Keras | Used in 6 of your 7 image-model practicals and in the lecturer's DR example, so your knowledge is visible | PyTorch (1 practical) |
| Compute | Train on a cloud GPU (Kaggle Notebooks or Colab); run the prototype on your Mac | EfficientNet fine-tuning is slow on CPU; Kaggle Notebooks can attach the dataset directly | Local Mac GPU (less predictable setup) |
| Dataset | APTOS-2019 (3,662 graded images, 5 stages) | Manageable in 16 days; same source as the lecturer's example; well documented | EyePACS-2015 (~35k images, patient IDs, but very large) |
| Resolution | Decide in Phase 6: 224×224 pre-resized copy vs original images | Tiny lesions (microaneurysms) need detail; originals allow our own cropping at higher resolution | Pre-resized copy (fastest) |
| Model | Pretrained EfficientNet (B0–B3) as the main candidate, MobileNetV2 as a fast baseline | Strong accuracy per compute; MobileNetV2 is already familiar | ResNet50, DenseNet121 |
| DR + stage | One 5-class model; "DR present" = 1 − P(No DR); report both levels | Simple, consistent, meets the brief | Two-output (DR yes/no + stage) model as an innovation option |
| Selection metric | Validation macro-F1 and QWK | Accuracy hides failure on rare stages (see #8) | Weighted-F1 |
| Experiments (4–5) | Baseline (frozen) → + enhancement → + balancing → + fine-tuning → backbone/resolution comparison | Each isolates one change that maps to a rubric row | Augmentation on/off (mirrors #6) |
| Prototype | Streamlit app: upload → preprocessing steps → DR yes/no + stage + probabilities → Grad-CAM → disclaimer | You have already built Streamlit apps | Gradio |

**Leakage guardrails:** remove duplicates before splitting; split once and save the split; augmentation and oversampling on training data only; class weights from training data only; the test set is used once at the end; thresholds and hyperparameters are tuned on validation only.

**Healthcare framing:** present the system as an academic screening prototype, not a diagnostic tool. Separate what the experiment shows from what would need clinical validation.

**Timeline (16 days)**

| Dates | Work |
|---|---|
| 17–18 Sep | Phases 2–5: requirement map, research, system design |
| 19–21 Sep | Phases 6–7: dataset, exploration, preprocessing |
| 21–23 Sep | Phases 8–10: augmentation/balancing, model, baseline training |
| 23–26 Sep | Phase 11: experiments |
| 26–27 Sep | Phase 12: final evaluation and error analysis |
| 27–29 Sep | Phases 13–14: prototype and evidence |
| 29 Sep – 2 Oct | Phases 15–16: report and video |
| 3 Oct | Buffer and submission |

**Open decisions for Mohamed**

1. GPU access: Kaggle Notebooks, Colab, or both? Which chip is in your Mac?
2. Framework: happy with TensorFlow/Keras?
3. Is "DR present + stage (0–4)" the right reading of the brief?
4. Any other lecture materials (e.g., classical image processing) to add as background?

---

## Appendix — the `Practical/` folder (added 19 September 2026)

A second folder, `Computer vision/Practical/`, holds the lecture practicals as they were run on Mohamed's own machine, plus the module's environment setup.

**Module environment**

- `setup_env.sh` (macOS) and `setup_env.bat` (Windows): delete any old `nibm_env`, create a fresh `python3 -m venv nibm_env`, upgrade pip, `pip install -r requirements.txt`, install `ipykernel`, and register a Jupyter kernel called "NIBM Bot".
- `requirements.txt`: tensorflow, numpy, pandas.
- `nibm_env/`: built from Anaconda's Python **3.13.9** (`/opt/anaconda3`). It contains TensorFlow **2.21.0**, Keras **3.14.1**, NumPy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, matplotlib 3.10.9, seaborn 0.13.2, h5py and Pillow — but no OpenCV, Streamlit or PyTorch.
- The copy in Downloads was originally created under `~/Personal/Lecture notes/…`, so it cannot be reused in place: a virtual environment records the absolute path it was built at.

**Practicals actually executed on this machine** (from execution counts and saved outputs)

| Notebook | Evidence |
|---|---|
| `01_Hand_written_Character_Recognition.ipynb` | Ran — MLP 97.4 %, CNN 99.2 % test accuracy |
| `Deep_Learning_Image_Classification_fixed.ipynb` | Ran — the student's own improved Fashion-MNIST version: L2 + BatchNorm + Dropout, `EarlyStopping(restore_best_weights=True)`, `validation_split=0.1`, train/validation curves; 85.4 % test accuracy |
| `fashion.ipynb` | Ran, but the saved run shows 10.2 % test accuracy: the cells were executed out of order, so an untrained model was evaluated |
| `diabetes_practical.ipynb` | Ran — 76.6 % test accuracy (tabular data) |
| `siamese_network_tutorial.ipynb` | **Ran fully on CPU** (50 epochs, ~60 s each) on the AT&T faces in `data/s1…s40`; best validation loss 0.119; test accuracy 83.75 % after tuning the distance threshold to 0.10; `best_model.h5` (14 MB) is that trained model |
| `Colab_Eye_Disease_Prediction.ipynb` | Only the first cell ran, failing with `No module named 'google.colab'` — the lecturer's DR notebook only works in Colab |
| `Defense_Multi_Agent_LLM_fast.ipynb` + `app.py` | A Streamlit app with cached resources, a progress bar and a two-column layout (LLM agents, not computer vision) |

**What this changes**

- **Environment:** the assignment environment follows the same venv + ipykernel recipe, built from the Anaconda Python already on the Mac, with OpenCV and Streamlit added.
- **Existing knowledge is stronger than the first pass suggested:** regularisation, early stopping and curve plotting are hands-on work, and a full training/evaluation pipeline (callbacks, classification report, confusion matrix, ROC, t-SNE, threshold tuning, model saving) has been run end to end on real images.
- **Two more lessons for our pipeline:** a metric can look like random guessing while the model is actually learning (the Siamese threshold at 0.5), and out-of-order notebook execution can produce a convincing but meaningless result (`fashion.ipynb`). Both argue for running notebooks top to bottom and saving metrics and models to files.
- **The classical image-processing gap is unchanged:** no CLAHE, sharpening, thresholding or morphology anywhere, and OpenCV is not in the module environment.

## Web references used

- Keras EfficientNet source docstring (input range 0–255, built-in rescaling): https://raw.githubusercontent.com/keras-team/keras/master/keras/src/applications/efficientnet.py
- Keras legacy `ImageDataGenerator` source (marked DEPRECATED): https://raw.githubusercontent.com/keras-team/keras/master/keras/src/legacy/preprocessing/image.py
- APTOS-2019 solution that removed duplicate training images: https://github.com/abhuse/aptos-retinopathy-detection
- APTOS 2019 Blindness Detection (Kaggle): https://www.kaggle.com/competitions/aptos2019-blindness-detection/data
