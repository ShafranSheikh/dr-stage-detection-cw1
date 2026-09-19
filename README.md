# Diabetic Retinopathy Stage Detection — Computer Vision CW1

**Author:** Mohamed Shafran · NIBM (BSCCOMP24.2P)
**Module:** Computer Vision · Coursework 1 · **Deadline:** 3 October 2026

Automatic grading of diabetic retinopathy (DR) from retinal fundus photographs.
For each image the system answers two questions:

1. Is diabetic retinopathy present? (Yes / No)
2. Which stage does it look like? (0 No DR, 1 Mild, 2 Moderate, 3 Severe, 4 Proliferative)

> **This is an academic prototype built for a university assignment. It is not a medical
> device and must not be used for diagnosis or treatment decisions.**

## Repository layout

| Folder | Contents |
|---|---|
| `kaggle_notebooks/` | Notebooks run on Kaggle: exploration, preprocessing, training, evaluation |
| `app/` | Streamlit prototype and the shared preprocessing code |
| `models/` | Trained model files (not tracked by Git — too large) |
| `results/` | Experiment log and metric tables |
| `report_figures/` | Figures used in the report |
| `report/` | Report drafts and the final PDF |
| `docs/` | Planning documents and the setup guide |

## Environment

- **Training:** Kaggle Notebooks with a GPU
- **Prototype:** macOS (Apple M2), Python 3.11, TensorFlow/Keras, Streamlit
- **Setup steps:** see `docs/SETUP.md`

## Data

APTOS 2019 Blindness Detection (Kaggle). The images are not stored in this repository.

## Originality

All code in this repository is written for this coursework. Standard libraries
(TensorFlow/Keras, OpenCV, scikit-learn) are used as tools; the pipeline, preprocessing,
experiments and application code are my own work.
