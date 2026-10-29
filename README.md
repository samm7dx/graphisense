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

- **Language & Runtime**: Python 3.10 or newer. The repository workflow has been checked on Python 3.13.15; other Python versions should be verified in their own environments.
- **Computer Vision & Image Processing**: OpenCV (`opencv-python`), NumPy
- **Machine Learning & Pipeline**: Scikit-Learn, SciPy
- **Deep Learning**: PyTorch, Torchvision (ResNet-18 transfer learning)
- **Data Engineering**: Pandas
- **Visualization**: Matplotlib, Plotly
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
│   │   ├── Potential Dysgraphia/ # Local class folder (git-ignored)
│   │   └── Low Potential Dysgraphia/ # Local class folder (git-ignored)
│   └── processed/               # Intermediate preprocessed data
├── notebooks/
│   ├── Stepwise-Guided-Notebook.ipynb  # Exploratory and guided analysis
│   └── inference_demo.ipynb            # End-to-end inference demonstration notebook
├── outputs/
│   ├── features/                # Extracted feature dataset (features.csv)
│   ├── figures/                 # Diagnostic plots, ROC curves, confusion matrices
│   ├── models/                  # Saved pipelines
│   │   ├── model_pipeline.joblib        # Random Forest pipeline
│   │   ├── model_metadata.json          # Random Forest metadata
│   │   └── cnn/                         # CNN/ResNet-18 checkpoints
│   │       ├── checkpoints/best_model.pt
│   │       └── checkpoints/final_model.pt
│   └── reports/                 # Statistical and model evaluation markdown reports
├── src/
│   ├── features/                # Static feature extraction and dataset builders
│   │   ├── handwriting_features.py  # 25 static feature extraction functions
│   │   ├── build_feature_dataset.py # Batch feature extraction pipeline
│   │   └── analyze_features.py      # Statistical analysis & hypothesis testing
│   ├── models/                  # Model training, cross-validation, and inference
│   │   ├── train_evaluate.py    # Stratified 5-fold CV training & evaluation (RF)
│   │   ├── inference.py         # Deterministic frozen-model inference engine (RF)
│   │   ├── cnn_dataset.py       # CNN dataset discovery, transforms, splits
│   │   ├── cnn_model.py         # ResNet-18 binary classifier
│   │   ├── cnn_train.py         # CNN training with early stopping & checkpoints
│   │   ├── cnn_inference.py     # CNN checkpoint-based inference engine
│   │   └── cnn_evaluate.py      # CNN held-out evaluation & metrics
│   ├── preprocessing/           # Image ingestion and segmentation
│   │   ├── image_loader.py      # Robust image reading and color conversion
│   │   └── binarization.py      # Polarity-aware Otsu/Triangle binarization
│   └── visualization/           # Plotting utilities and visual demos
├── tests/                       # Automated unit and integration tests
├── requirements.txt             # Runtime, test, and notebook dependencies
└── README.md                    # Project documentation and reproduction guide
```

---

## Architecture & Analysis Pipeline

GraphiSense executes an end-to-end, modular computer vision and machine learning pipeline with two analysis modes:

### Common Preprocessing (Both Modes)

1. **Image Loading & Grayscale Normalization** (`src/preprocessing/image_loader.py`):
   - Safely ingests standard image formats (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`).
   - Validates existence, regular file status, non-zero file sizes, and handles non-ASCII paths.
   - Normalizes single-channel, 3-channel, and 4-channel formats into a 2D uint8 grayscale array.
2. **Polarity-Aware Binarization** (`src/preprocessing/binarization.py`):
   - Correctly accommodates dark-background / light-stroke dataset crops without blind color inversion.
   - Auto-detects paper documents (light background / dark ink) vs. inverted crops via border margin intensity analysis.
   - Strictly enforces the standard mask convention: **Handwriting Foreground = 255 (white), Background = 0 (black)**.

### Mode A: Basic Analysis — Random Forest

3. **Static Feature Extraction** (`src/features/handwriting_features.py`):
   - Extracts 25 static geometric, topological, and stroke distribution measurements:
     - *Image Geometry*: Width, height, aspect ratio.
     - *Ink Density*: Proportion of active stroke pixels ($N_{\text{fg}} / (H \times W)$).
     - *Connected Components*: Speckle-filtered count, component width, height, and area statistics.
     - *Spatial Distribution*: Normalized horizontal/vertical centroids, row-projection distribution, active row ratio.
     - *Stroke Width Proxy*: Medial-axis Euclidean distance transform thickness proxy ($2 \times R_{\text{inscribed}}$).
4. **Model Training & Evaluation** (`src/models/train_evaluate.py`):
   - Evaluates Logistic Regression and Random Forest classifiers using Stratified 5-Fold Cross-Validation.
   - Uses `reduced_static_features` (9 spatial ratios and pixel-unit stroke thickness proxies) rather than raw crop dimensions for the primary model.
5. **Deterministic Single-Image Inference** (`src/models/inference.py`):
   - Loads the serialized Random Forest pipeline artifact (`outputs/models/model_pipeline.joblib`) and metadata schema (`outputs/models/model_metadata.json`).
   - Performs inference without fitting or altering model weights.
   - Provides model class-probability estimates mapped to `LPD` and `PD`.

### Mode B: Advanced Analysis — CNN / ResNet-18

3. **CNN Dataset & Transforms** (`src/models/cnn_dataset.py`):
   - Discovers images in nested dataset structures, maps class labels (`PD`/`LPD` → 1/0).
   - Applies ImageNet normalization and training augmentations (flip, rotation, color jitter).
   - Creates stratified train/validation/test splits with optional participant-level grouping.
4. **ResNet-18 Transfer Learning** (`src/models/cnn_model.py`):
   - Loads pretrained ImageNet weights, replaces final FC layer with binary classification head.
   - Configurable backbone freezing, dropout, and device placement (CPU/CUDA).
4. **Training & Checkpointing** (`src/models/cnn_train.py`):
   - AdamW optimizer, cross-entropy loss, ReduceLROnPlateau scheduler.
   - Early stopping on validation loss, saves best checkpoint with full metadata.
5. **Checkpoint-Based Inference** (`src/models/cnn_inference.py`):
   - Loads frozen checkpoint, applies ImageNet transforms, returns class probabilities.
6. **Held-Out Evaluation** (`src/models/cnn_evaluate.py`):
   - Evaluates on isolated test split using saved split metadata.
   - Reports accuracy, balanced accuracy, per-class precision/recall/F1, ROC-AUC, confusion matrix.

### Interactive Application Interface

6. **Streamlit Dashboard** (`app/main.py`):
   - Dual-mode interface: **Basic Analysis (Random Forest)** and **Advanced Analysis (CNN/ResNet-18)**.
   - Mode selector with real-time availability indicators.
   - Image upload, sample selection, preprocessing preview, and mode-specific results.
   - **Clean, expandable results layout**: Prediction output displayed first (class, probabilities, compact original/binary image previews). Secondary details (Feature Analysis, Model Architecture, Training History, Evaluation, Limitations) in collapsible expanders — no tabbed interface.
   - **Grouped feature distributions**: Violin plots organized by comparable scale (Ink Density & Normalized Geometry [0-1], Component Aspect Ratios, Stroke Width Statistics [px]) for clearer LPD vs PD comparison.
   - Research disclaimer and non-diagnostic notices prominently displayed.

---

## Installation & Setup (Windows PowerShell)

### 1. Prerequisites
Install Python 3.10 or newer. Use the Python launcher (`py`) to select the interpreter.

### 2. Environment Setup
Open Windows PowerShell in the repository root and create a project-local virtual environment:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, allow it for the current terminal only and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

The requirements include PyTorch and Torchvision (CPU versions by default). For CUDA support, install from the [PyTorch website](https://pytorch.org/get-started/locally/) matching your CUDA version.

### 3. Download and place the dataset

The dataset is not included in this repository. Download it from the [Mendeley Data record](https://data.mendeley.com/datasets/39hr8dx76p/1) and review the provider's terms before use. Then extract the downloaded ZIP so both class folders are directly inside the expected dataset directory:

```powershell
$datasetZip = Read-Host "Enter the full path to the downloaded Mendeley dataset ZIP"
$datasetRoot = Join-Path $PWD "data\raw"
if (-not (Test-Path -LiteralPath $datasetZip -PathType Leaf)) { throw "Dataset ZIP not found: $datasetZip" }
New-Item -ItemType Directory -Force -Path $datasetRoot
Expand-Archive -LiteralPath $datasetZip -DestinationPath $datasetRoot
Get-ChildItem -Path $datasetRoot
```

The resulting layout must be:

```text
data\raw\
├── Potential Dysgraphia\
└── Low Potential Dysgraphia\
```

If the ZIP contains an extra enclosing directory, move the two class folders so they sit directly under `$datasetRoot`. The raw dataset path is ignored by Git; do not add the dataset to commits.

### 4. Generate features and train the models

From the repository root with `.venv` activated, run:

**Random Forest (Basic Analysis):**
```powershell
python -m src.features.build_feature_dataset
python -m src.features.analyze_features
python -m src.models.train_evaluate
```

Feature extraction writes `outputs\features\features.csv`. The analysis and model-evaluation commands create the expected reports and figures. The model-evaluation command also trains the final Random Forest and saves:

```powershell
outputs\models\model_pipeline.joblib
outputs\models\model_metadata.json
```

Feature generation requires both class folders and supported images in each. Training and evaluation stop with an actionable error if the feature CSV is missing, contains failed rows, has missing or invalid feature values, or does not include both classes.

**CNN/ResNet-18 (Advanced Analysis):**
```powershell
python -m src.models.cnn_train --dataset-root data/raw --output-dir outputs/models/cnn --epochs 50 --batch-size 16 --num-workers 0
python -m src.models.cnn_evaluate --checkpoint outputs/models/cnn/checkpoints/best_model.pt --dataset-root data/raw --output-dir outputs/reports/cnn_evaluation
```

The CNN training command supports configurable epochs, batch size, learning rate, weight decay, split ratios, pretrained weights, backbone freezing, dropout, and early stopping. Run `python -m src.models.cnn_train --help` for all options. The best checkpoint is saved to `outputs\models\cnn\checkpoints\best_model.pt` with full metadata. Evaluation uses the saved split metadata to ensure the same held-out test set.

### 5. Run automated tests

```powershell
python -m pytest tests/ -q
```

Tests that require the external dataset or generated model/report artifacts are skipped when those files are unavailable. Synthetic preprocessing, input-validation, schema, and model tests can run without downloading the dataset.

### 6. Launch the web application

```powershell
python -m streamlit run app/main.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser. Uploads are limited to JPG/JPEG/PNG in the interface. The model must be trained as described above to complete predictions.

---

## Running Inference Programmatically

You can evaluate any handwriting image directly via the Python API:

### Random Forest (Basic Analysis)
```python
from pathlib import Path
from src.models.inference import predict_handwriting

# Provide path to an image file (.jpg, .png, etc.)
image_file = Path("data/raw/Potential Dysgraphia/PD (1).jpg")

# Execute deterministic inference
result = predict_handwriting(image_file)

print("Predicted Label:", result["predicted_label"])
print("Class Probabilities:", result["class_probabilities"])
print("Key Features Used:", result["features_used"])
```

### CNN/ResNet-18 (Advanced Analysis)
```python
from pathlib import Path
from src.models.cnn_inference import predict_handwriting_cnn

# Provide path to an image file (.jpg, .png, etc.)
image_file = Path("data/raw/Potential Dysgraphia/PD (1).jpg")

# Execute CNN inference (loads checkpoint automatically)
result = predict_handwriting_cnn(image_file)

print("Predicted Label:", result["predicted_label"])
print("Class Probabilities:", result["class_probabilities"])
print("Model:", result.get("model_type", "ResNet-18"))
print("Checkpoint:", result.get("checkpoint", "default"))
```

Both functions return a dictionary with the predicted label, class probabilities, model metadata, and a non-diagnostic disclaimer. The CNN function loads the default checkpoint from `outputs/models/cnn/checkpoints/best_model.pt` unless overridden.

---

## Dataset Attribution and Organization

- **Source**: Mendeley Data, *Potential Dysgraphia Handwriting Dataset of School-Age Children*, Ramlan et al. (Oct 2023), DOI: [10.17632/39hr8dx76p.1](https://data.mendeley.com/datasets/39hr8dx76p.1). Follow the provider's license and attribution requirements.
- **Placement**: Place the class folders under:
  ```text
  data\raw\
  ├── Potential Dysgraphia/
  │   └── ...
  └── Low Potential Dysgraphia/
      └── ...
  ```

Raw handwriting images and generated outputs are excluded from Git by `.gitignore`. This project is a research screening-support prototype, not a diagnostic tool.
