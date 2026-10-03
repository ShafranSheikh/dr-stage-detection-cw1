"""
dr_gradcam.py - Grad-CAM heatmaps for the DR model.

It is used in two places, so the heatmaps are made in exactly the same way:
  * Kaggle notebook 4 (dr-04-evaluate), for the error analysis;
  * the Streamlit prototype (app/), to explain the prediction for an uploaded photo.

Grad-CAM (Selvaraju et al., 2017) shows which parts of an image pushed the model towards one stage:
  1. take the feature maps of the last convolution layer (7 x 7 maps for a 224 px input);
  2. measure how much the stage's score would change if each map grew a little (its gradient);
  3. average those gradients over each map -> one importance weight per map;
  4. add the maps up with those weights and keep only the positive part (evidence FOR the stage);
  5. scale to 0-1 and enlarge to the image size, so it can be laid over the photo.
The score used is the stage's value before softmax (its "logit"), as in the original paper.

Computer Vision CW1 - Diabetic Retinopathy Stage Detection - Mohamed Shafran
"""
import numpy as np
import cv2
import keras
import tensorflow as tf

# the last convolution block of each backbone (the names Keras gives them)
LAST_CONV = {"EfficientNetB0": "top_activation", "EfficientNetB3": "top_activation",
             "MobileNetV2": "out_relu"}


class GradCam:
    """Grad-CAM for a model built by dr_training.build_model:
    image -> augmentation -> backbone -> 'gap' -> 'dropout' -> 'stage_probs' (Dense, softmax)."""

    def __init__(self, model, last_conv):
        self.model = model
        # one pass gives both the last feature maps and the pooled features that feed the output layer
        self.inner = keras.Model(model.inputs,
                                 [model.get_layer(last_conv).output, model.get_layer("gap").output])
        output_layer = model.get_layer("stage_probs")
        self.kernel, self.bias = output_layer.kernel, output_layer.bias

    def maps(self, images, stages):
        """Low-resolution heatmaps (N x h x w, each scaled to 0-1) for a batch of 0-255 images
        and, for each image, the stage to explain. Also returns the five probabilities."""
        x = tf.convert_to_tensor(np.asarray(images, dtype="float32"))
        stages = np.asarray(stages, dtype="int32")
        with tf.GradientTape() as tape:
            features_maps, pooled = self.inner(x, training=False)       # training=False: no augmentation
            logits = keras.ops.matmul(pooled, self.kernel) + self.bias  # scores before softmax
            score = tf.gather(logits, stages, axis=1, batch_dims=1)     # each image's own stage
        grads = tape.gradient(score, features_maps)                     # N x h x w x channels
        weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)     # step 3: one weight per map
        cams = tf.nn.relu(tf.reduce_sum(features_maps * weights, axis=-1)).numpy()   # step 4
        peak = cams.reshape(len(cams), -1).max(axis=1)
        cams = cams / np.maximum(peak, 1e-12)[:, None, None]            # step 5: 0-1 per image
        probs = keras.ops.convert_to_numpy(keras.ops.softmax(logits, axis=-1))
        return cams, probs

    def heatmap(self, image, stage, size=None):
        """One image: the heatmap enlarged to `size` x `size` (default: the image's own size)."""
        cams, _ = self.maps(image[None], [stage])
        return enlarge(cams[0], size or image.shape[0])


def enlarge(cam, size):
    """Smoothly enlarge a low-resolution heatmap (e.g. 7 x 7) to size x size."""
    return np.clip(cv2.resize(cam.astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC), 0, 1)


def overlay(image, cam, alpha=0.4):
    """Colour the heatmap (blue = low, red = high) and blend it over the image (RGB uint8)."""
    if cam.shape[:2] != image.shape[:2]:
        cam = enlarge(cam, image.shape[0])
    colour = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return np.uint8(np.clip((1 - alpha) * image.astype(np.float32) + alpha * colour, 0, 255))
