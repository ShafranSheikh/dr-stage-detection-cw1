"""
app.py - the prototype. Upload a fundus photograph and see how the system answers it.

Run it from the project folder, with the virtual environment active:

    source venv/bin/activate
    streamlit run app/app.py

The screen is all this file does. The preprocessing comes from dr_preprocessing.py (notebook 2),
the heatmaps from dr_gradcam.py (notebook 4), and the answer from dr_predict.py - the same code
the Kaggle notebooks used, so the app shows exactly what the model was trained to see.

Computer Vision CW1 - Diabetic Retinopathy Stage Detection - Mohamed Shafran
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                                   # draw to images, never to a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))    # so the shared modules can be imported
import dr_preprocessing as prep
import dr_gradcam as gradcam
import dr_predict as engine

APP_DIR = Path(__file__).resolve().parent
DEMO_DIR = APP_DIR / "demo_images"
MODEL_DIR = engine.MODEL_DIR
CLASS_NAMES = engine.CLASS_NAMES
STAGE_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
INK_SOFT, MUTED, GRID = "#52514e", "#898781", "#e1e0d9"

st.set_page_config(page_title="Diabetic Retinopathy Stage Detection", layout="wide")


# ---------------------------------------------------------------------------
# Loading (cached, so the model is read from disk only once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the model...")
def get_model():
    return engine.load_model()


@st.cache_data
def demo_photos():
    """The example photographs in app/demo_images, with their true stage if labels.csv is there."""
    if not DEMO_DIR.exists():
        return {}
    labels = {}
    if (DEMO_DIR / "labels.csv").exists():
        table = pd.read_csv(DEMO_DIR / "labels.csv")
        labels = dict(zip(table["id_code"], table["diagnosis"]))
    return {path.name: labels.get(path.stem) for path in sorted(DEMO_DIR.glob("*.png"))}


@st.cache_data
def test_scores():
    """Notebook 4's headline test results, if they have been copied into models/."""
    path = MODEL_DIR / "test_metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


def probability_chart(probabilities, answered_stage):
    """A bar per stage; the stage the system answered is outlined."""
    fig, ax = plt.subplots(figsize=(6.2, 2.6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    y = np.arange(5)
    bars = ax.barh(y, probabilities * 100, height=0.62, color=STAGE_COLORS)
    bars[answered_stage].set_edgecolor("#0b0b0b")
    bars[answered_stage].set_linewidth(1.6)
    for yi, value in zip(y, probabilities * 100):
        ax.text(value + 1.5, yi, f"{value:.0f}%", va="center", fontsize=9, color=INK_SOFT)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{k} {name}" for k, name in enumerate(CLASS_NAMES)], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xticks([])
    ax.tick_params(colors=MUTED)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("Diabetic Retinopathy Stage Detection")
st.caption("Computer Vision CW1 · Mohamed Shafran · NIBM (BSCCOMP24.2P)")
st.warning("**Academic prototype.** It was trained on the public APTOS 2019 dataset for a university "
           "assignment. It is not a medical device, has had no clinical validation, and must not be "
           "used to diagnose anyone or to make decisions about any person's care.")

try:
    model, config, cam = get_model()
except Exception as error:                                  # missing files, or a version problem
    st.error(f"The model could not be loaded.\n\n`{error}`\n\nPut **best_model.keras** and "
             f"**best_config.json** from notebook 3's Kaggle output into `{MODEL_DIR}`.")
    st.stop()

with st.sidebar:
    st.subheader("The model")
    st.markdown(
        f"""
- **Network:** {config['backbone']}, pretrained on ImageNet
- **Input:** {config['image_size']} × {config['image_size']} px, variant {config['variant']}
  ({prep.VARIANTS[config['variant']]})
- **Class balancing:** {config['balance']}
- **Fine-tuning:** top {config['unfreeze_fraction']:.0%} of the network, learning rate
  {config['fine_tune_lr']:g}
- **Chosen by:** best validation macro-F1 across the experiments in notebook 3
"""
    )
    scores = test_scores()
    if scores:
        test = scores["results"]["final model (test)"]
        st.subheader("Measured on the test split")
        st.markdown(
            f"""
- Stage accuracy **{test['accuracy']:.0%}**, macro-F1 **{test['macro_f1']:.2f}**, QWK **{test['qwk']:.2f}**
- DR yes/no: sensitivity **{test['dr_sensitivity']:.0%}**, specificity **{test['dr_specificity']:.0%}**
- {scores['test_images']} photographs, used once, never during training or model choice
"""
        )
    else:
        st.caption(f"Validation macro-F1 {config['macro_f1']:.2f}, QWK {config['qwk']:.2f} "
                   "(notebook 3). Copy `test_metrics.json` into models/ to show the test results here.")

    with st.expander("How it works"):
        st.markdown(
            """
1. **Find the retina** and crop to it, so the black border is dropped.
2. **Give every photo the same window** — the full retina width, with the top and bottom trimmed
   flat — because some cameras cut the retina off and the camera itself hints at the diagnosis.
3. **Resize and enhance** exactly as in training.
4. **The network** returns a probability for each of the five stages.
5. **Two-step answer:** DR is present if stages 1–4 together reach 50 %; the stage is then the most
   likely of stages 1–4.
6. **Grad-CAM** colours the areas that pushed the network towards that stage.
"""
        )
    st.caption("The stages are the international scale: 0 No DR, 1 Mild, 2 Moderate, 3 Severe, "
               "4 Proliferative.")

# --- choose a photograph ----------------------------------------------------
examples = demo_photos()
source = "Upload a photograph"
if examples:
    source = st.radio("Photograph", ["Upload a photograph", "Use an example"], horizontal=True,
                      label_visibility="collapsed")

photo_bytes, photo_name, true_stage = None, None, None
if source == "Upload a photograph":
    upload = st.file_uploader("Choose a fundus photograph (PNG or JPEG)", type=["png", "jpg", "jpeg"])
    if upload is not None:
        photo_bytes, photo_name = upload.getvalue(), upload.name
else:
    chosen = st.selectbox("Example photographs (from the test split, never used in training)",
                          list(examples))
    photo_bytes, photo_name = (DEMO_DIR / chosen).read_bytes(), chosen
    true_stage = examples[chosen]

if photo_bytes is None:
    st.info("Choose a photograph to start. A fundus photograph is a picture of the inside back of "
            "the eye, taken through the pupil.")
    st.stop()

# --- run the system ---------------------------------------------------------
try:
    rgb = prep.load_rgb(photo_bytes)
except ValueError:
    st.error("That file could not be read as an image.")
    st.stop()

with st.spinner("Preparing the photograph and running the model..."):
    out = engine.predict(rgb, model, config, cam)

if len(out["problems"]) + len(out["unusual"]) >= 3:      # brightness, contrast, sharpness, size, shape
    st.error("**Three or more checks flagged this photograph**, so it is very unlike the ones the "
             "model was trained on — it may not be a fundus photograph at all. The model still "
             "produces an answer below, but that answer should not be trusted. (Only about one "
             "training photograph in 500 trips three checks.)")

# --- 1. what the system sees -------------------------------------------------
st.subheader("1. The photograph, and what the model sees")
steps = st.columns(3 if config["variant"] != "A" else 2)
steps[0].image(rgb, caption=f"{photo_name} — as taken ({rgb.shape[1]} × {rgb.shape[0]} px)",
               width="stretch")
steps[1].image(out["standard"], caption="Retina cropped, resized, same window for every photo",
               width="stretch")
if config["variant"] != "A":
    steps[2].image(out["variant_image"],
                   caption=f"Variant {config['variant']}: {prep.VARIANTS[config['variant']]}",
                   width="stretch")

# --- 2. the answer -----------------------------------------------------------
st.subheader("2. The system's answer")
answer, chart = st.columns([1, 2], gap="large")
with answer:
    st.metric("Diabetic retinopathy", "Present" if out["dr_present"] else "Not present",
              border=True)
    st.metric(f"Stage {out['stage']}", out["stage_name"], border=True)
    st.caption(f"Probability of DR (stages 1–4 added up): **{out['p_dr']:.0%}** · "
               f"probability of stage {out['stage']}: **{out['stage_probability']:.0%}**")
    if true_stage is not None:
        mark = "matches" if int(true_stage) == out["stage"] else "differs from"
        st.caption(f"The dataset's label for this photograph is **{int(true_stage)} "
                   f"{CLASS_NAMES[int(true_stage)]}** — the answer {mark} it.")
with chart:
    st.pyplot(probability_chart(out["probabilities"], out["stage"]))

for note in out["problems"]:                     # reasons to trust the answer less
    st.warning(note)
for note in out["unusual"]:                      # different from the training photos, but not worse
    st.caption(note)

# --- 3. where the model looked ----------------------------------------------
st.subheader("3. Where the model looked (Grad-CAM)")
heat, explanation = st.columns([1, 1], gap="large")
if out["heatmap"] is not None:
    heat.image(gradcam.overlay(out["standard"], out["heatmap"]),
               caption=f"Red = the areas that pushed the model towards stage {out['stage']}",
               width="stretch")
# EfficientNet rounds UP at each stride, so 300 px gives a 10 x 10 feature map, not 300 // 32 = 9.
grid = -(-config["image_size"] // 32)
cell = config["image_size"] / grid
explanation.markdown(
    f"""
The heatmap comes from the last convolution layer of the network, which sees the photograph as a
grid of {grid} × {grid} patches. Each patch covers
roughly {cell:.0f} × {cell:.0f} pixels of the model's input, so the colour shows a **region**, not an exact lesion,
and the hottest point can sit next to what the model actually used.

For an answer of *No DR* the map shows what the model relied on to call the eye healthy, which
rarely points at anything in particular.

Grad-CAM shows **where** the model looked, not **why**. A sensible-looking heatmap does not make
the answer correct, and an odd-looking one does not make it wrong.
"""
)

with st.expander("Technical details"):
    st.markdown(
        f"""
| Item | Value |
|---|---|
| Retina found | radius {out['info']['radius_px']:.0f} px, height/width {out['info']['box_ratio']:.2f} |
| Share of the visible retina kept | {out['info']['fov_kept']:.2f} |
| Enlarged to reach 512 px | {"yes" if out['info']['upscaled'] else "no"} |
| Brightness / contrast / sharpness | {out['measures']['brightness']:.0f} / {out['measures']['contrast']:.1f} / {out['measures']['sharpness']:.0f} |
| Middle 90 % of the training photographs | {engine.TYPICAL_RANGE['brightness'][0]:.0f}–{engine.TYPICAL_RANGE['brightness'][1]:.0f} / {engine.TYPICAL_RANGE['contrast'][0]:.1f}–{engine.TYPICAL_RANGE['contrast'][1]:.1f} / {engine.TYPICAL_RANGE['sharpness'][0]:.0f}–{engine.TYPICAL_RANGE['sharpness'][1]:.0f} |
| Flagged outside (1st–99th percentile) | {engine.FLAG_RANGE['brightness'][0]:.0f}–{engine.FLAG_RANGE['brightness'][1]:.0f} / {engine.FLAG_RANGE['contrast'][0]:.1f}–{engine.FLAG_RANGE['contrast'][1]:.1f} / {engine.FLAG_RANGE['sharpness'][0]:.0f}–{engine.FLAG_RANGE['sharpness'][1]:.0f} |
| Probabilities | {", ".join(f"{CLASS_NAMES[k]} {p:.3f}" for k, p in enumerate(out['probabilities']))} |
| Decision rule | {config['decision_rule']} |
"""
    )

st.divider()
st.caption("Trained on APTOS 2019 photographs from a small number of cameras and one region. On a "
           "photograph from a different camera or population the answer may be much less reliable. "
           "This prototype is coursework, not a diagnosis.")
