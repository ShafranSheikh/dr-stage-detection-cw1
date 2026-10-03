"""
dr_predict.py - everything needed to answer "DR? which stage?" for ONE photograph.

app.py is only the screen; all the deciding happens here, so it can be tested without a browser:

    python3 dr_predict.py ../app/demo_images/901a3552fe26.png

That command also prints the library versions and is the quickest way to check that the model
saved on Kaggle loads on this Mac.

The photo goes through exactly the steps the training images went through, using the same two
files as the Kaggle notebooks: dr_preprocessing.py (notebook 2) and dr_gradcam.py (notebook 4).

Computer Vision CW1 - Diabetic Retinopathy Stage Detection - Mohamed Shafran
"""
import json
from pathlib import Path

import cv2
import numpy as np

import dr_preprocessing as prep
import dr_gradcam as gradcam

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR.parent / "models"          # best_model.keras and best_config.json go here
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

# What the TRAINING photographs looked like, measured on the standardised image (variant A) in
# notebook 2 (results/notebook2_v1/preprocess_stats.csv, 2,441 training images).
#   TYPICAL_RANGE - the middle 90 % (5th-95th percentile). Shown for context only.
#   FLAG_RANGE    - the 1st-99th percentile: outside this, a photograph really is unusual for this
#                   model. 6 % of the training photographs themselves fall outside it, against 26 %
#                   for the middle-90 % range, which is why the wider range decides the warnings.
TYPICAL_RANGE = {"brightness": (50.3, 120.0), "contrast": (9.6, 27.8), "sharpness": (8.5, 55.3)}
FLAG_RANGE = {"brightness": (40.9, 133.4), "contrast": (7.8, 34.3), "sharpness": (6.0, 68.9)}
MIN_RADIUS_PX = 275                            # 1st percentile of the training retina radius, in pixels


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_model(model_dir=MODEL_DIR):
    """Load the model notebook 3 saved, its settings, and a Grad-CAM helper for it.
    Returns (model, config, cam). Raises FileNotFoundError with a clear message if a file is missing."""
    import keras                               # imported here so this file stays quick to import

    model_dir = Path(model_dir)
    for name in ("best_model.keras", "best_config.json"):
        if not (model_dir / name).exists():
            raise FileNotFoundError(f"{name} is missing from {model_dir} - download it from "
                                    "notebook 3's output on Kaggle")
    config = json.loads((model_dir / "best_config.json").read_text())
    model = keras.models.load_model(model_dir / "best_model.keras", compile=False)   # no training here
    cam = gradcam.GradCam(model, gradcam.LAST_CONV[config["backbone"]])
    return model, config, cam


# ---------------------------------------------------------------------------
# One photograph
# ---------------------------------------------------------------------------
def prepare(rgb, config):
    """The three pictures the app shows, plus what was measured while cropping:
    the standardised retina (variant A), the enhancement variant the model was trained on,
    and that variant resized to the model's input size."""
    standard, info = prep.standardise(rgb)                    # crop to the retina, resize, common window
    variant = config["variant"]
    if variant == "B":
        shown = prep.clahe_unsharp(standard)                  # contrast + edge enhancement
    elif variant == "C":
        shown = prep.ben_graham(standard)                     # local colour normalisation
    else:
        shown = standard
    size = int(config["image_size"])
    model_input = shown if size == prep.SIZE else cv2.resize(shown, (size, size),
                                                             interpolation=cv2.INTER_AREA)
    return standard, shown, model_input, info


def quality_notes(standard, info):
    """Compare this photograph with the training photographs. Returns the measurements and two lists:
    `problems` - reasons to trust the answer less (too dark, flat, blurred, or too little retina);
    `unusual`  - ways the photograph is unlike the training set without being worse (very sharp,
                 very contrasty). Both empty means nothing stood out."""
    stats = prep.image_stats(standard)
    measured = {
        "brightness": 0.299 * stats["mean_r"] + 0.587 * stats["mean_g"] + 0.114 * stats["mean_b"],
        "contrast": stats["contrast"],
        "sharpness": stats["sharpness"],
    }
    low_word = {"brightness": "darker", "contrast": "flatter", "sharpness": "blurrier"}
    high_word = {"brightness": "brighter", "contrast": "more contrasty", "sharpness": "sharper"}
    problems, unusual = [], []
    for name, value in measured.items():
        low, high = FLAG_RANGE[name]
        typical_low, typical_high = TYPICAL_RANGE[name]
        context = (f"({name} {value:.0f}; the middle 90 % of the training photographs are "
                   f"{typical_low:.0f}-{typical_high:.0f})")
        if value < low:
            problems.append(f"This photograph is {low_word[name]} than almost every training "
                            f"photograph {context}, so the answer may be less reliable.")
        elif value > high:
            unusual.append(f"This photograph is {high_word[name]} than almost every training "
                           f"photograph {context}. That is not a fault, but the model saw few like it.")
    if info["radius_px"] < MIN_RADIUS_PX:
        problems.append(f"The retina is only {2 * info['radius_px']:.0f} pixels across, smaller than "
                        "almost every training photograph, so fine detail is missing.")
    if info["box_ratio"] < 0.55:
        problems.append("Only a narrow band of the retina is visible, so the window may miss lesions "
                        "above and below it.")
    return measured, problems, unusual


def decide(probabilities):
    """The two-step rule fixed in the design (the same rule as dr_training.two_step):
    DR is present if the probabilities of stages 1-4 add up to 0.5 or more; the stage is then the
    most likely of stages 1-4, otherwise stage 0."""
    probabilities = np.asarray(probabilities, dtype=float)
    p_dr = float(probabilities[1:].sum())
    stage = 1 + int(np.argmax(probabilities[1:])) if p_dr >= 0.5 else 0
    return {"dr_present": bool(p_dr >= 0.5), "p_dr": p_dr, "stage": stage,
            "stage_name": CLASS_NAMES[stage], "stage_probability": float(probabilities[stage])}


def predict(rgb, model, config, cam=None):
    """Everything the app shows for one photograph, in one dictionary."""
    standard, shown, model_input, info = prepare(rgb, config)
    probabilities = model.predict(model_input[None].astype("float32"), verbose=0)[0]
    result = decide(probabilities)
    measured, problems, unusual = quality_notes(standard, info)
    heatmap = None
    if cam is not None:                                       # where the model looked, at display size
        low_res, _ = cam.maps(model_input[None], [result["stage"]])
        heatmap = gradcam.enlarge(low_res[0], prep.SIZE)
    return {"standard": standard, "variant_image": shown, "model_input": model_input,
            "probabilities": np.asarray(probabilities, dtype=float), "heatmap": heatmap,
            "measures": measured, "problems": problems, "unusual": unusual, "info": info, **result}


# ---------------------------------------------------------------------------
# Command line: also the quickest check that this machine can run the model
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import time

    import keras
    import tensorflow as tf

    print(f"python {sys.version.split()[0]} | tensorflow {tf.__version__} | keras {keras.__version__} "
          f"| opencv {cv2.__version__} | numpy {np.__version__}")
    model, config, cam = load_model()
    print(f"model loaded: {config['name']} ({config['backbone']} at {config['image_size']} px, "
          f"variant {config['variant']}) - {model.count_params() / 1e6:.1f} M parameters")

    if len(sys.argv) < 2:
        print("\nto try a photo:  python3 dr_predict.py <photo.png>")
        sys.exit(0)

    for path in sys.argv[1:]:
        start = time.time()
        out = predict(prep.load_rgb(path), model, config, cam)
        print(f"\n{Path(path).name}  ({time.time() - start:.1f} s)")
        print(f"  DR present: {'YES' if out['dr_present'] else 'no'} ({out['p_dr']:.0%}) | "
              f"stage {out['stage']} {out['stage_name']} ({out['stage_probability']:.0%})")
        print("  probabilities: " + ", ".join(f"{CLASS_NAMES[k]} {p:.2f}"
                                              for k, p in enumerate(out["probabilities"])))
        for note in out["problems"]:
            print(f"  warning: {note}")
        for note in out["unusual"]:
            print(f"  note: {note}")
