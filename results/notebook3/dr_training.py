"""
dr_training.py - trains ONE configuration and scores it on the validation split.

Notebook 3 runs this file once per training run, each time in a fresh Python process:
    python dr_training.py '{"name": "E1_A_s42", "backbone": "EfficientNetB0", ...}'
Why a fresh process: TensorFlow does not always hand back all the memory of a model that has
been deleted, so thirteen runs inside one process would pile it up. A new process starts
clean, and a crash in one run cannot damage the results of the others.

The notebook also imports this file for its shared pieces (augmentation, model, metrics),
so the figures and tables are made with exactly the same code as the training.

Computer Vision CW1 - Diabetic Retinopathy Stage Detection - Mohamed Shafran
"""
import os
import sys
import json
import time

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

tf.get_logger().setLevel("ERROR")                        # hide TensorFlow's retracing notices

# ---------------------------------------------------------------------------
# Settings (notebook 3, section 1 explains each one)
# ---------------------------------------------------------------------------
WEIGHTS = "imagenet"
SIZES = {"EfficientNetB0": 224, "EfficientNetB3": 300, "MobileNetV2": 224}
BATCH = {"EfficientNetB0": 32, "EfficientNetB3": 16, "MobileNetV2": 32}
HEAD_EPOCHS, HEAD_LR, HEAD_PATIENCE = 8, 1e-3, 3           # phase 1: backbone frozen
FT_EPOCHS, FT_LR, FT_PATIENCE = 20, 1e-4, 5               # phase 2: top of the backbone fine-tuned
UNFREEZE_FRACTION = 0.25
DROPOUT = 0.3

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
BLUE, ORANGE, INK, INK_SOFT, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


# ---------------------------------------------------------------------------
# Augmentation: Keras layers that are active ONLY while training
# ---------------------------------------------------------------------------
def make_augmenter(fill):
    """Random flips, any rotation, zoom, shift, brightness and contrast.
    `fill` is the background value for areas uncovered by rotating or shifting."""
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.5, fill_mode="constant", fill_value=fill),        # 0.5 turn = +-180 deg
        layers.RandomZoom(0.1, fill_mode="constant", fill_value=fill),
        layers.RandomTranslation(0.05, 0.05, fill_mode="constant", fill_value=fill),
        layers.RandomBrightness(0.15, value_range=(0, 255)),
        layers.RandomContrast(0.15, value_range=(0, 255)),
    ], name="augment")


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
def build_model(backbone, size, fill, weights="default"):
    """Input (0-255 RGB) -> augmentation -> pretrained backbone -> GAP -> dropout -> 5 probabilities.
    The backbone is built on the augmentation's output rather than nested as one layer, so its
    last convolution layer stays reachable for Grad-CAM. Returns the model, the backbone's own
    layers (for freezing) and the name of its last convolution layer."""
    weights = WEIGHTS if weights == "default" else weights     # None = random start (tests only)
    inputs = keras.Input((size, size, 3), name="image")
    x = make_augmenter(fill)(inputs)
    if backbone == "MobileNetV2":                        # MobileNetV2 expects -1..1, EfficientNet 0..255
        x = layers.Rescaling(1 / 127.5, offset=-1.0, name="to_minus1_plus1")(x)
        base = keras.applications.MobileNetV2(include_top=False, weights=weights, input_tensor=x)
        last_conv = "out_relu"
    else:
        application = getattr(keras.applications, backbone)          # EfficientNetB0 / EfficientNetB3
        base = application(include_top=False, weights=weights, input_tensor=x)
        last_conv = "top_activation"
    y = layers.GlobalAveragePooling2D(name="gap")(base.output)
    y = layers.Dropout(DROPOUT, name="dropout")(y)
    outputs = layers.Dense(5, activation="softmax", name="stage_probs")(y)
    model = keras.Model(inputs, outputs, name=f"dr_{backbone}")

    ours = {"image", "augment", "to_minus1_plus1"}                    # layers before the backbone
    backbone_layers = [layer for layer in base.layers if layer.name not in ours]
    return model, backbone_layers, last_conv


def set_trainable(backbone_layers, top_fraction):
    """Freeze the backbone, then unfreeze its top `top_fraction` of layers - except
    BatchNormalization, which stays frozen (inference mode) so its ImageNet statistics survive."""
    first_unfrozen = len(backbone_layers) - int(round(len(backbone_layers) * top_fraction))
    for i, layer in enumerate(backbone_layers):
        layer.trainable = i >= first_unfrozen and not isinstance(layer, layers.BatchNormalization)


# ---------------------------------------------------------------------------
# Data and class balancing (training split only)
# ---------------------------------------------------------------------------
def class_weights(y_train):
    """'balanced' weights from the TRAINING labels: n_images / (5 x n_images_of_that_stage)."""
    return dict(enumerate(compute_class_weight("balanced", classes=np.arange(5), y=y_train)))


def make_train_dataset(X, y, idx, batch, balance, seed):
    """tf.data pipeline for the training images: shuffle (or oversample), batch, cast, prefetch.
    Returns the dataset and the number of batches in one epoch."""
    def as_float(images, labels):
        return tf.cast(images, tf.float32), labels
    if balance == "oversample":                          # one endless stream per stage, drawn equally
        streams = []
        for stage in range(5):
            own = idx[y[idx] == stage]
            streams.append(tf.data.Dataset.from_tensor_slices((X[own], y[own]))
                           .shuffle(len(own), seed=seed).repeat())
        ds = tf.data.Dataset.sample_from_datasets(streams, weights=[0.2] * 5, seed=seed)
    else:
        ds = tf.data.Dataset.from_tensor_slices((X[idx], y[idx])).shuffle(len(idx), seed=seed).repeat()
    steps = int(np.ceil(len(idx) / batch))
    ds = ds.batch(batch).map(as_float, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    return ds, steps


# ---------------------------------------------------------------------------
# The system's answer and the metrics
# ---------------------------------------------------------------------------
def two_step(probs):
    """DR present if P(stages 1-4) >= 0.5; then the most likely of stages 1-4; otherwise stage 0."""
    has_dr = probs[:, 1:].sum(axis=1) >= 0.5
    return np.where(has_dr, 1 + probs[:, 1:].argmax(axis=1), 0)


def scores(true, pred):
    """Metrics for the 5 stages and for DR yes/no (the same definitions as notebook 2)."""
    true, pred = np.asarray(true), np.asarray(pred)
    dr_true, dr_pred = true > 0, pred > 0
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, labels=list(range(5)), average="macro", zero_division=0)),
        "qwk": float(cohen_kappa_score(true, pred, weights="quadratic")),
        "dr_accuracy": float(accuracy_score(dr_true, dr_pred)),
        "dr_sensitivity": float(recall_score(dr_true, dr_pred, zero_division=0)),
        "dr_specificity": float(recall_score(~dr_true, ~dr_pred, zero_division=0)),
    }


class ValMetrics(keras.callbacks.Callback):
    """After every epoch: validation macro-F1, QWK and DR accuracy of the two-step answer, added to
    the logs so that the callbacks listed after this one (checkpoint, early stopping, learning-rate
    schedule, CSV log) can use them."""
    def __init__(self, X_val, y_val, batch_size):
        super().__init__()
        self.X_val, self.y_val, self.batch_size = X_val, y_val, batch_size

    def on_epoch_end(self, epoch, logs=None):
        probs = self.model.predict(self.X_val, batch_size=self.batch_size, verbose=0)
        s = scores(self.y_val, two_step(probs))
        logs["val_macro_f1"], logs["val_qwk"], logs["val_dr_accuracy"] = s["macro_f1"], s["qwk"], s["dr_accuracy"]
        print(f"   epoch {epoch + 1:>2}: loss {logs['loss']:.3f} | val loss {logs['val_loss']:.3f} | "
              f"val macro-F1 {s['macro_f1']:.3f} | QWK {s['qwk']:.3f} | DR acc {s['dr_accuracy']:.3f}",
              flush=True)


def plot_curves(history, split_epoch, title, path, show=False):
    """Training vs validation accuracy and loss over both phases; dashed line = fine-tuning starts."""
    epochs = np.arange(1, len(history) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="#fcfcfb")
    for ax, key, label in ((axes[0], "accuracy", "Accuracy"), (axes[1], "loss", "Loss")):
        ax.set_facecolor("#fcfcfb")
        ax.plot(epochs, history[key], color=BLUE, linewidth=2, label="training")
        ax.plot(epochs, history[f"val_{key}"], color=ORANGE, linewidth=2, label="validation")
        ax.axvline(split_epoch + 0.5, color=MUTED, linestyle="--", linewidth=1)
        ax.text(split_epoch + 0.6, ax.get_ylim()[1], " fine-tuning", va="top", fontsize=8.5, color=INK_SOFT)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))   # whole epochs only
        ax.ticklabel_format(axis="y", useOffset=False)
        ax.set_xlabel("Epoch", color=INK_SOFT)
        ax.set_ylabel(label, color=INK_SOFT)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.legend(frameon=False, labelcolor=INK_SOFT)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle(title, fontsize=12, fontweight="bold", color=INK)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(path, dpi=110, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# One training run
# ---------------------------------------------------------------------------
def train_run(cfg, X, splits, smoke=False):
    """Phase 1 (backbone frozen) then phase 2 (top of the backbone fine-tuned); reload the best
    checkpoint of both phases, score it on the validation split, save everything in run_dir.
    `smoke=True` shrinks the run to a few images and two batches per phase."""
    run_dir = cfg["run_dir"]
    os.makedirs(run_dir, exist_ok=True)
    keras.utils.set_random_seed(cfg["seed"])
    start = time.time()

    y = splits["diagnosis"].to_numpy()
    idx_train = np.where(splits["split"] == "train")[0]
    idx_val = np.where(splits["split"] == "val")[0]
    if smoke:                                            # 12 images per stage, 24 validation images
        idx_train = np.concatenate([idx_train[y[idx_train] == s][:12] for s in range(5)])
        idx_val = idx_val[:24]
    size, batch = SIZES[cfg["backbone"]], BATCH[cfg["backbone"]]
    X_val, y_val = X[idx_val].astype("float32"), y[idx_val]
    fill = 128.0 if cfg["variant"] == "C" else 0.0       # background of variant C is grey, else black

    ft_lr = float(cfg.get("ft_lr", FT_LR))               # experiment E4 also tries 1e-5
    model, backbone_layers, _ = build_model(cfg["backbone"], size, fill)
    train_ds, epoch_steps = make_train_dataset(X, y, idx_train, batch, cfg["balance"], cfg["seed"])
    weights = class_weights(y[idx_train]) if cfg["balance"] == "weights" else None
    checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(run_dir, "best.keras"),
                                                 monitor="val_macro_f1", mode="max",
                                                 save_best_only=True)      # shared by both phases

    def fit(epochs, learning_rate, patience, phase):
        model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        callbacks = [ValMetrics(X_val, y_val, batch), checkpoint,
                     keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max",
                                                   patience=patience, restore_best_weights=True),
                     keras.callbacks.ReduceLROnPlateau(monitor="val_macro_f1", mode="max",
                                                       factor=0.5, patience=2, min_lr=1e-7),
                     keras.callbacks.CSVLogger(os.path.join(run_dir, f"history_phase{phase}.csv"))]
        return model.fit(train_ds, steps_per_epoch=2 if smoke else epoch_steps, epochs=epochs,
                         validation_data=(X_val, y_val), class_weight=weights,
                         callbacks=callbacks, verbose=0)

    print(f"--- {cfg['name']}: {cfg['backbone']} {size} px, variant {cfg['variant']}, "
          f"balance {cfg['balance']}, fine-tuning lr {ft_lr:g}, seed {cfg['seed']}", flush=True)
    print(" phase 1 - backbone frozen, head learning", flush=True)
    set_trainable(backbone_layers, 0.0)
    h1 = fit(1 if smoke else HEAD_EPOCHS, HEAD_LR, HEAD_PATIENCE, 1)
    print(" phase 2 - top 25% of the backbone fine-tuned", flush=True)
    set_trainable(backbone_layers, UNFREEZE_FRACTION)
    h2 = fit(1 if smoke else FT_EPOCHS, ft_lr, FT_PATIENCE, 2)

    history = pd.concat([pd.DataFrame(h1.history), pd.DataFrame(h2.history)], ignore_index=True)
    history.insert(0, "phase", [1] * len(h1.history["loss"]) + [2] * len(h2.history["loss"]))
    best = keras.models.load_model(os.path.join(run_dir, "best.keras"))
    probs = best.predict(X_val, batch_size=batch, verbose=0)
    answer = two_step(probs)

    result = {**cfg, "size": size, "ft_lr": ft_lr, **scores(y_val, answer),
              "macro_f1_argmax": float(f1_score(y_val, probs.argmax(axis=1), labels=list(range(5)),
                                                average="macro", zero_division=0)),
              "phase1_best_macro_f1": float(max(h1.history["val_macro_f1"])),
              "best_epoch": int(np.argmax(history["val_macro_f1"].to_numpy()) + 1),
              "epochs_phase1": len(h1.history["loss"]), "epochs_phase2": len(h2.history["loss"]),
              "minutes": round((time.time() - start) / 60, 1)}
    mixed = splits.iloc[idx_val]["camera_group"].to_numpy() == cfg.get("mixed_camera")
    if mixed.sum() >= 5:                                 # the camera that gives the least away
        mixed_scores = scores(y_val[mixed], answer[mixed])
        result["mixed_camera_macro_f1"] = mixed_scores["macro_f1"]
        result["mixed_camera_dr_accuracy"] = mixed_scores["dr_accuracy"]
    if not smoke:
        history.to_csv(os.path.join(run_dir, "history.csv"), index=False)
        pd.DataFrame({"id_code": splits.iloc[idx_val]["id_code"].to_numpy(), "true": y_val, "pred": answer,
                      **{f"p{k}": probs[:, k] for k in range(5)}}
                     ).to_csv(os.path.join(run_dir, "val_predictions.csv"), index=False)
        plot_curves(history, len(h1.history["loss"]), cfg["name"], os.path.join(run_dir, "curves.png"))
    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f" => val macro-F1 {result['macro_f1']:.3f} | QWK {result['qwk']:.3f} | "
          f"DR accuracy {result['dr_accuracy']:.3f} | {result['minutes']} min", flush=True)
    return result


def main(cfg_json):
    """Entry point for `python dr_training.py '<json>'`: load the split and the cached images, train."""
    matplotlib.use("Agg")                                # a background process draws to files only
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    cfg = json.loads(cfg_json)
    splits = pd.read_csv(cfg["splits_csv"])
    X = np.load(cfg["cache"])                            # uint8 array, rows in splits.csv order
    train_run(cfg, X, splits, smoke=cfg.get("smoke", False))


if __name__ == "__main__":
    main(sys.argv[1])
