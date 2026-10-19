# GraphiSense

**Computer Vision-Based Digital Handwriting Analysis for Potential Dysgraphia Screening Support**

GraphiSense is an experimental academic research prototype designed to investigate whether static handwriting morphology and geometric stroke features can provide screening support for potential dysgraphia indicators.

---

## Important Scientific & Ethical Safeguards

1. **Screening Support, Not Clinical Diagnosis**: GraphiSense is an academic research prototype. Outputs and probability scores are statistical estimates on a specific experimental dataset; they **do not constitute medical, clinical, psychiatric, or educational diagnoses**, nor do they replace licensed multidisciplinary clinical assessments.
2. **Dataset Label Nature**: Dataset labels (`PD` for Potential Dysgraphia, `LPD` for Low Potential Dysgraphia) represent research dataset categories from the original data collector, not independently verified clinical diagnoses.
3. **Static vs. Dynamic Limitations**: Static 2D scanned/photographed handwriting images **cannot measure true physical pen-tip pressure, writing velocity, acceleration, or in-air cognitive hesitation pauses**. All features extracted here (such as distance-transform stroke thickness and ink density) are purely 2D morphological and spatial proxies.
4. **Image-Level Validation Caveat**: The underlying dataset does not provide participant or writer identifiers. Consequently, cross-validation was partitioned at the **image level**. If multiple line crops originate from the same writer, samples from the same writer may occur across training and validation folds, which can yield optimistic performance estimates compared to true writer-independent evaluation. No writer IDs are invented or assumed.
5. **Generalization**: Predictions on training or validation samples must not be presented as evidence of clinical generalization to unconstrained handwriting or diverse classroom populations.

---

## Technologies & Stack

- **Language & Runtime**: Python 3.10+ (tested on Python 3.10 and 3.14)
- **Computer Vision & Image Processing**: OpenCV (`opencv-python`), NumPy
- **Machine Learning & Pipeline**: Scikit-Learn, SciPy
- **Data Engineering**: Pandas
- **Visualization**: Matplotlib
- **Web Application Interface**: Streamlit
- **Testing & Verification**: Pytest

---

## Repository Structure

```text
graphisense/
├── app/
│   ├── main.py                  # Streamlit interactive web application
│   └── pipeline_handler.py      # Decoupled application pipeline handler
├── data/
│   ├── raw/                     # Raw image storage (dataset excluded via .gitignore)
│   │   ├── README.md            # Dataset placement instructions
│   │   └── handwriting/         # Local dataset extraction directory (git-ignored)
│   └── processed/               # Intermediate preprocessed data
├── notebooks/
│   ├── Stepwise-Guided-Notebook.ipynb  # Exploratory and guided analysis
│   └── inference_demo.ipynb            # End-to-end inference demonstration notebook
├── outputs/
│   ├── features/                # Extracted feature dataset (features.csv)
│   ├── figures/                 # Diagnostic plots, ROC curves, confusion matrices
│   ├── models/                  # Saved pipeline (model_pipeline.joblib, model_metadata.json)
│   └── reports/                 # Statistical and model evaluation markdown reports
├── src/
│   ├── features/                # Static feature extraction and dataset builders
│   │   ├── handwriting_features.py  # 25 static feature extraction functions
│   │   ├── build_feature_dataset.py # Batch feature extraction pipeline
│   │   └── analyze_features.py      # Statistical analysis & hypothesis testing
│   ├── models/                  # Model training, cross-validation, and inference
│   │   ├── train_evaluate.py    # Stratified 5-fold CV training & evaluation
│   │   └── inference.py         # Deterministic frozen-model inference engine
│   ├── preprocessing/           # Image ingestion and segmentation
│   │   ├── image_loader.py      # Robust image reading and color conversion
│   │   └── binarization.py      # Polarity-aware Otsu/Triangle binarization
│   └── visualization/           # Plotting utilities and visual demos
├── tests/                       # Automated test suite (51 unit & integration tests)
├── requirements.txt             # Minimal, reproducible dependencies
└── README.md                    # Project documentation and reproduction guide
```

---

## Architecture & Analysis Pipeline

GraphiSense executes an end-to-end, modular computer vision and machine learning pipeline:

1. **Image Loading & Grayscale Normalization** (`src/preprocessing/image_loader.py`):
   - Safely ingests standard image formats (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`).
   - Validates existence, regular file status, non-zero file sizes, and handles non-ASCII paths.
   - Normalizes single-channel, 3-channel, and 4-channel formats into a 2D uint8 grayscale array.
2. **Polarity-Aware Binarization** (`src/preprocessing/binarization.py`):
   - Correctly accommodates dark-background / light-stroke dataset crops without blind color inversion.
   - Auto-detects paper documents (light background / dark ink) vs. inverted crops via border margin intensity analysis.
   - Strictly enforces the standard mask convention: **Handwriting Foreground = 255 (white), Background = 0 (black)**.
3. **Static Feature Extraction** (`src/features/handwriting_features.py`):
   - Extracts 25 static geometric, topological, and stroke distribution measurements:
     - *Image Geometry*: Width, height, aspect ratio.
     - *Ink Density*: Proportion of active stroke pixels ($N_{\text{fg}} / (H \times W)$).
     - *Connected Components*: Speckle-filtered count, component width, height, and area statistics.
     - *Spatial Distribution*: Normalized horizontal/vertical centroids, row-projection distribution, active row ratio.
     - *Stroke Width Proxy*: Medial-axis Euclidean distance transform thickness proxy ($2 \times R_{\text{inscribed}}$).
4. **Model Training & Evaluation** (`src/models/train_evaluate.py`):
   - Evaluates Logistic Regression and Random Forest classifiers using Stratified 5-Fold Cross-Validation.
   - Leverages `reduced_static_features` (9 scale-invariant spatial ratios and stroke thickness proxies) to avoid confounding from arbitrary image crop dimensions.
5. **Deterministic Single-Image Inference** (`src/models/inference.py`):
   - Loads the serialized Random Forest pipeline artifact (`outputs/models/model_pipeline.joblib`) and metadata schema (`outputs/models/model_metadata.json`).
   - Performs inference without fitting or altering model weights.
   - Provides calibrated class probability distribution mapped to `LPD` and `PD`.
6. **Interactive Application Interface** (`app/main.py`):
   - Streamlit dashboard providing image upload and verified dataset sample inspection.
   - Displays side-by-side original and binarized masks, detected polarity, primary metrics, probability distributions, and full metadata schemas.

---

## Installation & Setup (Windows PowerShell)

### 1. Prerequisites
Ensure Python 3.10, 3.11, 3.12, or 3.14 is installed on your system.

### 2. Environment Setup
Open Windows PowerShell and navigate to the project directory:

```powershell
# Navigate to the project directory
cd <path-to-graphisense>

# Create a virtual environment (optional if using an existing one)
python -m venv .venv

# Activate the virtual environment in PowerShell
.\.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install required dependencies
python -m pip install -r requirements.txt
```

### 3. Verify Imports
Verify that all core libraries import without error:

```powershell
python -c "import cv2, numpy, pandas, matplotlib, scipy, sklearn, streamlit, pytest; print('All required libraries imported successfully!')"
```

---

## Running Automated Tests

Run the full automated test suite using `pytest`:

```powershell
# Run complete test suite with detailed output
python -m pytest tests/ -v

# Run tests with summary report
python -m pytest tests/ -q
```

The test suite covers:
- Image loading, channel normalization, and invalid file handling (`tests/test_image_loader.py`)
- Polarity detection, thresholding methods, and mask invariants (`tests/test_binarization.py`)
- Synthetic geometric verification, noise component filtering, and CSV integrity (`tests/test_features.py`)
- Statistical distribution analysis, Cliff's delta, and FDR bounds (`tests/test_analysis.py`)
- Model pipeline construction, CV reproducibility, and metric bounds (`tests/test_models.py`)
- Deterministic inference, metadata schema validation, and corrupt artifact handling (`tests/test_inference.py`)
- Application pipeline handler, byte decoding, polarity modes, and error paths (`tests/test_app.py`)

---

## Launching the Web Application

Launch the local Streamlit application:

```powershell
streamlit run app/main.py
```

Or invoke via Python module:

```powershell
python -m streamlit run app/main.py
```

Once launched, open your web browser at `http://localhost:8501`.

---

## Running Inference Programmatically

You can evaluate any handwriting image directly via the Python API:

```python
from pathlib import Path
from src.models.inference import predict_handwriting

# Provide path to an image file (.jpg, .png, etc.)
image_file = Path("data/raw/handwriting/extracted/DATASET DYSGRAPHIA HANDWRITING/Potential Dysgraphia/PD (1).jpg")

# Execute deterministic inference
result = predict_handwriting(image_file)

print("Predicted Label:", result["predicted_label"])
print("Class Probabilities:", result["class_probabilities"])
print("Key Features Used:", result["features_used"])
```

---

## Dataset Organization & Setup

- **Source**: Mendeley Data (*Potential Dysgraphia Handwriting Dataset of School-Age Children*, Ramlan et al., Oct 2023, DOI: [10.17632/39hr8dx76p.1](https://data.mendeley.com/datasets/39hr8dx76p.1)).
- **Size**: 249 images (114 Potential Dysgraphia `PD`, 135 Low Potential Dysgraphia `LPD`).
- **Placement**: Extract the downloaded dataset into:
  ```text
  data/raw/handwriting/extracted/DATASET DYSGRAPHIA HANDWRITING/
  ├── Potential Dysgraphia/
  │   ├── PD (1).jpg
  │   └── ...
  └── Low Potential Dysgraphia/
      ├── LPD (1).jpg
      └── ...
  ```

Note: Raw handwriting dataset images are excluded from Git version control via `.gitignore`.
