"""
dr_preprocessing.py - the ONE preprocessing file of this project.

It is used in two places, so every image is prepared in exactly the same way:
  * Kaggle notebook 2 (dr-02-preprocess), to build the cached training images;
  * the Streamlit prototype (app/), to prepare a photo a user uploads.

Every function works on a single image and uses only that image's own pixels.
Nothing is learned from other images, so nothing can leak between the
train, validation and test sets.

Three variants are produced from the same standardised image:
  A  basic          - retina centred, resized, same window shape for every photo
  B  CLAHE + unsharp - A + contrast adjustment + edge enhancement
  C  Ben Graham      - A + local colour normalisation (Graham, 2015)

Computer Vision CW1 - Diabetic Retinopathy Stage Detection - Mohamed Shafran
"""
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Settings (each one is explained in notebook 2)
# ---------------------------------------------------------------------------
SIZE = 512              # every output image is SIZE x SIZE pixels
MASK_SCALE = 0.95       # keep 95% of the radius: drops the dark rim at the edge of the lens
FOV_BAND = 0.74         # keep only rows within 0.74 x radius of the centre - the level at which
                        # the most cut-off cameras end - so every photo gets the same shape
BG_TOL = 10             # grey level at or below which a pixel counts as black background
ROW_FRACTION = 0.02     # a row or column is part of the retina if >2% of its pixels are bright

CLAHE_CLIP = 2.0        # CLAHE contrast limit: higher = stronger boost, but more noise
CLAHE_TILES = (8, 8)    # CLAHE equalises each tile of an 8 x 8 grid separately
UNSHARP_SIGMA = 2.0     # size (in pixels) of the fine detail the unsharp mask sharpens
UNSHARP_AMOUNT = 1.0    # how much of that detail is added back (1.0 = detail doubled)
BEN_GRAHAM_SIGMA = SIZE / 2 / 30                 # radius / 30, as in Graham (2015): ~8.5 px

VARIANTS = {"A": "basic", "B": "CLAHE + unsharp mask", "C": "Ben Graham"}
PNG_SETTINGS = [cv2.IMWRITE_PNG_COMPRESSION, 3]   # lossless; 3 = good balance of size and speed


def window_mask(size=SIZE, scale=MASK_SCALE, band=FOV_BAND):
    """The field of view every image is given: a centred disc with its top and bottom
    cut off flat. `scale` shrinks the disc and the band together."""
    centre = (size - 1) / 2
    yy, xx = np.ogrid[:size, :size]
    disc = (xx - centre) ** 2 + (yy - centre) ** 2 <= (scale * size / 2) ** 2
    rows = np.abs(yy - centre) <= scale * band * size / 2
    return disc & rows


MASK = window_mask()                            # the same window for every image
INNER = window_mask(SIZE, MASK_SCALE * 0.9)     # a smaller window, away from the mask edge


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def load_rgb(source):
    """Read an image from a file path or from raw bytes (an upload) and return RGB uint8."""
    if isinstance(source, (bytes, bytearray)):
        data = np.frombuffer(source, np.uint8)
    else:
        data = np.fromfile(str(source), np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)       # also drops an alpha channel
    if bgr is None:
        raise ValueError("the file could not be read as an image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Steps 1-3: the same field of view for every photo  ->  variant A
# ---------------------------------------------------------------------------
def retina_box(rgb, tol=BG_TOL):
    """Edges (top, bottom, left, right) of the retina and how many pixels it covers.
    A row or column counts only if more than ROW_FRACTION of it is bright, so a few
    stray bright pixels in the black border cannot stretch the box."""
    bright = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) > tol
    rows = np.where(bright.mean(axis=1) > ROW_FRACTION)[0]
    cols = np.where(bright.mean(axis=0) > ROW_FRACTION)[0]
    if len(rows) == 0 or len(cols) == 0:
        return None, 0                                # nothing bright in the whole image
    return (int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])), int(bright.sum())


def find_retina(rgb):
    """Centre and radius of the retina, plus its height/width ratio.
    The radius is half the retina's WIDTH: cameras that cut the retina off do so at the
    top and bottom, so the width is the full diameter. The ratio says how much was cut
    (1.0 = a complete circle)."""
    box, retina_pixels = retina_box(rgb)
    height, width = rgb.shape[:2]
    if box is None:                                   # fall back to the whole frame
        return (width - 1) / 2, (height - 1) / 2, min(height, width) / 2, height * width, 1.0
    top, bottom, left, right = box
    box_height, box_width = bottom - top + 1, right - left + 1
    return ((left + right) / 2, (top + bottom) / 2, box_width / 2, retina_pixels,
            box_height / box_width)


def standardise(rgb):
    """Variant A: a square around the retina, resized to SIZE x SIZE, with everything
    outside the common window blacked out. Returns the image and a dict describing it."""
    cx, cy, radius, retina_pixels, box_ratio = find_retina(rgb)
    side = max(int(round(2 * radius)), 1)
    x0, y0 = int(round(cx - radius)), int(round(cy - radius))

    height, width = rgb.shape[:2]
    pad = max(0, -x0, -y0, x0 + side - width, y0 + side - height)
    if pad:                                           # the square reaches past the photo's edge
        rgb = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        x0, y0 = x0 + pad, y0 + pad
    square = rgb[y0:y0 + side, x0:x0 + side]

    # INTER_AREA averages each block of pixels when shrinking (this also removes sensor
    # noise); INTER_CUBIC is the better choice for the rare photo smaller than SIZE.
    shrinking = side >= SIZE
    img = cv2.resize(square, (SIZE, SIZE),
                     interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC)
    img[~MASK] = 0

    window_area = MASK.sum() * (radius / (SIZE / 2)) ** 2      # in original pixels
    return img, {"radius_px": float(radius), "box_ratio": float(box_ratio),
                 "fov_kept": float(min(window_area / max(retina_pixels, 1), 1.0)),
                 "upscaled": not shrinking}


# ---------------------------------------------------------------------------
# Enhancement
# ---------------------------------------------------------------------------
def masked_blur(img, sigma, mask=MASK):
    """Gaussian blur that averages ONLY retina pixels (a 'normalised convolution').
    An ordinary blur would mix the black background into the edge of the retina and
    create a false bright ring there once the blur is subtracted."""
    weight = mask.astype(np.float32)
    total = cv2.GaussianBlur(img.astype(np.float32) * weight[..., None], (0, 0), sigma)
    norm = cv2.GaussianBlur(weight, (0, 0), sigma)[..., None]
    return total / np.maximum(norm, 1e-6)


def clahe_unsharp(img_a, return_steps=False):
    """Variant B: contrast adjustment (CLAHE) followed by edge enhancement (unsharp mask)."""
    # 1. CLAHE on the lightness channel of LAB only, so the colours stay natural
    lightness, a, b = cv2.split(cv2.cvtColor(img_a, cv2.COLOR_RGB2LAB))
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILES)
    contrast = cv2.cvtColor(cv2.merge([clahe.apply(lightness), a, b]), cv2.COLOR_LAB2RGB)
    contrast[~MASK] = 0

    # 2. unsharp mask: (image - blurred image) is the fine detail; add it back once more
    detail = contrast.astype(np.float32) - masked_blur(contrast, UNSHARP_SIGMA)
    sharp = np.clip(contrast + UNSHARP_AMOUNT * detail, 0, 255).astype(np.uint8)
    sharp[~MASK] = 0
    return (contrast, sharp) if return_steps else sharp


def ben_graham(img_a):
    """Variant C: subtract the local average colour, multiply by 4, centre on grey 128.
    Slow changes (uneven lighting, each camera's colour cast) disappear;
    local detail (vessels, lesions) stays."""
    local_average = masked_blur(img_a, BEN_GRAHAM_SIGMA)
    out = np.clip(4.0 * (img_a.astype(np.float32) - local_average) + 128.0, 0, 255).astype(np.uint8)
    out[~MASK] = 128                                  # neutral grey outside, as in the original
    return out


# ---------------------------------------------------------------------------
# One call for the app, one for building the cache
# ---------------------------------------------------------------------------
def preprocess(rgb, variant="A"):
    """The full pipeline for one image and one variant ('A', 'B' or 'C')."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; use one of {list(VARIANTS)}")
    img_a, _ = standardise(rgb)
    if variant == "B":
        return clahe_unsharp(img_a)
    if variant == "C":
        return ben_graham(img_a)
    return img_a


def all_variants(rgb):
    """All three variants from a single read of the photo (used to build the cache)."""
    img_a, info = standardise(rgb)
    return {"A": img_a, "B": clahe_unsharp(img_a), "C": ben_graham(img_a)}, info


def image_stats(img):
    """Numbers describing one processed image, measured inside the circle only."""
    pixels = img[MASK].astype(np.float32)
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(grey, cv2.CV_32F)
    means, spreads = pixels.mean(axis=0), pixels.std(axis=0)
    return {"mean_r": float(means[0]), "mean_g": float(means[1]), "mean_b": float(means[2]),
            "std_r": float(spreads[0]), "std_g": float(spreads[1]), "std_b": float(spreads[2]),
            "contrast": float(grey[MASK].std()),
            "sharpness": float(laplacian[INNER].var())}
