# GraphiSense

**Computer Vision-Based Digital Handwriting Analysis for Potential Dysgraphia Screening Support**

GraphiSense is an experimental academic research prototype designed to investigate whether static handwriting morphology and geometric stroke features can provide screening support for potential dysgraphia indicators.

---

## Table of Contents

- [Important Scientific & Ethical Safeguards](#important-scientific--ethical-safeguards)
- [Technologies & Stack](#technologies--stack)
- [Repository Structure](#repository-structure)
- [Architecture & Analysis Pipeline](#architecture--analysis-pipeline)
- [Quick Start](#quick-start)
- [Installation & Setup](#installation--setup)
- [Running the Application](#running-the-application)
- [Running Inference Programmatically](#running-inference-programmatically)
- [Dataset Attribution](#dataset-attribution)
- [Troubleshooting](#troubleshooting)
- [Testing](#testing)
- [Contributing](#contributing)

---

## Important Scientific & Ethical Safeguards

1. **Screening Support, Not Clinical Diagnosis**: GraphiSense is an academic research prototype. Outputs and probability scores are statistical estimates on a specific experimental dataset; they **do not constitute medical, clinical, psychiatric, or educational diagnoses**, nor do they replace licensed multidisciplinary clinical assessments.

2. **Dataset Label Nature**: Dataset labels (`PD` for Potential Dysgraphia, `LPD` for Low Potential Dysgraphia) represent research dataset categories from the original data collector, not independently verified clinical diagnoses.

3. **Static vs. Dynamic Limitations**: Static 2D scanned/photographed handwriting images **cannot measure true physical pen-tip pressure, writing velocity, acceleration, or in-air cognitive hesitation pauses**. All features extracted here (such as distance-transform stroke thickness and ink density) are purely 2D morphological and spatial proxies.

4. **Image-Level Validation Caveat**: The underlying dataset does not provide participant or writer identifiers. Consequently, cross-validation was partitioned at the **image level**. If multiple line crops originate from the same writer, samples from the same writer may occur across training and validation folds, which can yield optimistic performance estimates compared to true writer-independent evaluation. No writer IDs are invented or assumed.

5. **Generalization**: Predictions on training or validation samples must not be presented as evidence of clinical generalization to unconstrained handwriting or diverse classroom populations.

---

## Technologies & Stack

| Category | Technologies |
|----------|--------------|
| **Language & Runtime** | Python 3.10+ (tested on 3.10–3.13) |
| **Computer Vision** | OpenCV (`opencv-python`), NumPy |
| **Machine Learning** | Scikit-Learn, SciPy |
| **Deep Learning** | PyTorch, Torchvision (ResNet-18 transfer learning) |
| **Data Engineering** | Pandas |
| **Visualization** | Matplotlib, Plotly |
| **Web Interface** | Streamlit |
| **Testing** | Pytest |

---

## Repository Structure

```text
graphisense/
├── app/
│   ├── main.py                  # Streamlit application entry point & router
│   ├── pipeline_handler.py      # Decoupled preprocessing/inference pipeline
│   ├── components/              # Reusable UI components
│   │   ├── charts.py            # Plotly charts (gauges, violin, radar, etc.)
│   │   ├── model_details.py     # Feature analysis, model details, limitations
│   │   ├── navigation.py        # Sidebar navigation (6 sections)
│   │   ├── prediction_results.py # Prediction card, probability gauges, images
│   │   ├── preprocessing.py     # Preprocessing stage grid viewer
│   │   └── upload.py            # Image upload & sample selection
│   ├── page_renderers/          # Page-level renderers
│   │   ├── modelling.py         # Main workspace (upload → analyze → results)
│   │   ├── methodology.py       # Methodology, RF vs CNN comparison
│   │   ├── about.py             # About Dysgraphia
│   │   ├── preprocessing.py     # Preprocessing pipeline explorer
│   │   ├── repository_info.py   # Repository structure & run guide
│   │   └── comparison.py        # RF vs CNN comparison wrapper
│   └── services/                # Business logic & data loading
│       ├── inference_service.py # RF/CNN inference orchestration
│       ├── metrics_loader.py    # Evaluation metrics loading
│       ├── model_service.py     # Model availability & metadata
│       ├── feature_service.py   # Feature dataset loading
│       └── preprocessing_service.py # Pipeline stage computation
├── data/
│   ├── raw/                     # Raw images (git-ignored)
│   │   ├── README.md            # Dataset placement instructions
│   │   ├── Potential Dysgraphia/
│   │   └── Low Potential Dysgraphia/
│   └── processed/               # Intermediate preprocessed data
├── notebooks/
│   ├── Stepwise-Guided-Notebook.ipynb
│   └── inference_demo.ipynb
├── outputs/
│   ├── features/                # features.csv (249×29)
│   ├── figures/                 # ROC, confusion matrices, feature plots
│   ├── models/
│   │   ├── model_pipeline.joblib
│   │   ├── model_metadata.json
│   │   └── cnn/checkpoints/best_model.pt
│   └── reports/                 # Markdown evaluation reports
├── src/
│   ├── features/                # Static feature extraction
│   ├── models/                  # RF & CNN training/inference
│   ├── preprocessing/           # Image loading & binarization
│   └── visualization/           # Plotting utilities
├── tests/                       # Pytest suite (51 tests)
├── requirements.txt
└── README.md
```

---

## Architecture & Analysis Pipeline

GraphiSense executes a modular CV/ML pipeline with two analysis modes:

### Common Preprocessing (Both Modes)

1. **Image Loading & Grayscale Normalization** (`src/preprocessing/image_loader.py`):
   - Ingests standard formats (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`)
   - Validates existence, file status, non-zero size, handles non-ASCII paths
   - Normalizes to 2D uint8 grayscale

2. **Polarity-Aware Binarization** (`src/preprocessing/binarization.py`):
   - Handles dark-background/light-stroke dataset crops correctly
   - Auto-detects paper documents (light bg/dark ink) via border analysis
   - Enforces: **Handwriting = 255 (white), Background = 0 (black)**

### Mode A: Basic Analysis — Random Forest

3. **Static Feature Extraction** (`src/features/handwriting_features.py`):
   - 25 geometric, topological, stroke-distribution features:
     - Image Geometry: width, height, aspect ratio
     - Ink Density: $N_{\text{fg}} / (H \times W)$
     - Connected Components: count, width/height/area statistics
     - Spatial Distribution: normalized centroids, row projections, active row ratio
     - Stroke Width Proxy: medial-axis distance transform ($2 \times R_{\text{inscribed}}$)

4. **Model Training & Evaluation** (`src/models/train_evaluate.py`):
   - Logistic Regression & Random Forest with Stratified 5-Fold CV
   - Primary model uses 9 reduced static features (ratios + pixel proxies)

5. **Deterministic Single-Image Inference** (`src/models/inference.py`):
   - Loads frozen pipeline + metadata, no refitting
   - Returns class probabilities mapped to `LPD`/`PD`

### Mode B: Advanced Analysis — CNN / ResNet-18

3. **CNN Dataset & Transforms** (`src/models/cnn_dataset.py`):
   - Discovers nested dataset structures, maps `PD`/`LPD` → 1/0
   - ImageNet normalization + augmentations (flip, rotation, color jitter)
   - Stratified train/val/test splits

4. **ResNet-18 Transfer Learning** (`src/models/cnn_model.py`):
   - Pretrained ImageNet weights, binary classification head
   - Configurable backbone freezing, dropout, device placement

5. **Training & Checkpointing** (`src/models/cnn_train.py`):
   - AdamW, cross-entropy, ReduceLROnPlateau
   - Early stopping on val loss, saves best checkpoint with metadata

6. **Checkpoint-Based Inference** (`src/models/cnn_inference.py`):
   - Loads frozen checkpoint, applies ImageNet transforms, returns probabilities

7. **Held-Out Evaluation** (`src/models/cnn_evaluate.py`):
   - Evaluates on isolated test split using saved metadata
   - Reports accuracy, balanced accuracy, per-class P/R/F1, ROC-AUC, CM

### Interactive Application Interface (`app/main.py`)

- **Dual-mode interface**: Basic (RF) / Advanced (CNN/ResNet-18)
- **Real-time model availability** indicators
- **Clean, expandable results layout**:
  - Prediction output first: class, probabilities, compact original/binary previews (320px)
  - Secondary details in collapsible expanders — **no tabbed interface**
- **RF expanders**: Feature Analysis, Image Processing, Preprocessing Explorer, Model Details, Limitations
- **CNN expanders**: Image Processing, Model Architecture, Training History, Evaluation, Limitations
- **Grouped feature distributions**: Violin plots by comparable scale
  - Ink Density & Normalized Geometry [0–1]
  - Component Aspect Ratios
  - Stroke Width Statistics [px]
- **Compact preprocessing grid**: 3-column responsive layout with stage thumbnails
- Research disclaimers prominently displayed

---

## Quick Start

```powershell
# 1. Clone & enter repo
git clone https://github.com/samm7dx/graphisense.git
cd graphisense

# 2. Create venv & install deps
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3. Download dataset from Mendeley Data (DOI: 10.17632/39hr8dx76p.1)
#    Extract so class folders sit directly under data\raw\

# 4. Train Random Forest
python -m src.features.build_feature_dataset
python -m src.models.train_evaluate

# 5. (Optional) Train CNN
python -m src.models.cnn_train --dataset-root data/raw --output-dir outputs/models/cnn --epochs 50

# 6. Launch app
python -m streamlit run app/main.py
# Open http://localhost:8501
```

---

## Installation & Setup

### Prerequisites
- Python 3.10 or newer (use `py -3` launcher on Windows)
- Git
- ~2 GB disk space for dataset + outputs

### Environment Setup (Windows PowerShell)

```powershell
# Create & activate virtual environment
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip & install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> **Note**: If PowerShell blocks script activation, run once:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .\.venv\Scripts\Activate.ps1
> ```

Requirements include PyTorch/Torchvision (CPU by default). For CUDA, install from [pytorch.org](https://pytorch.org/get-started/locally/).

### Dataset Placement

Download the [Mendeley Potential Dysgraphia Handwriting Dataset](https://data.mendeley.com/datasets/39hr8dx76p/1) and extract so the structure is:

```text
data\raw\
├── Potential Dysgraphia\
│   ├── PD (1).jpg
│   └── ...
└── Low Potential Dysgraphia\
    ├── LPD (1).jpg
    └── ...
```

The `data\raw` directory is git-ignored — do not commit dataset files.

---

## Running the Application

```powershell
# From repo root with .venv activated
python -m streamlit run app/main.py
```

Open **http://localhost:8501** in your browser.

**UI Navigation (6 sections via sidebar):**
1. 🏠 **Modelling** — Upload/sample → Analyze → Results (default landing)
2. 🔬 **Methodology** — Pipeline details, feature descriptions, evaluation protocols
3. ⚖️ **RF vs CNN Comparison** — Side-by-side metrics (CV vs held-out)
4. ⚙️ **Preprocessing** — 6-stage pipeline grid with full-size viewer
5. 📁 **Repository Info** — Structure, artifacts, run commands
6. ℹ️ **About Dysgraphia** — Condition overview, prevalence, disclaimer

---

## Running Inference Programmatically

### Random Forest (Basic Analysis)

```python
from pathlib import Path
from src.models.inference import predict_handwriting

image_file = Path("data/raw/Potential Dysgraphia/PD (1).jpg")
result = predict_handwriting(image_file)

print("Predicted Label:", result["predicted_label"])
print("Class Probabilities:", result["class_probabilities"])
print("Key Features Used:", result["features_used"])
```

### CNN/ResNet-18 (Advanced Analysis)

```python
from pathlib import Path
from src.models.cnn_inference import predict_handwriting_cnn

image_file = Path("data/raw/Potential Dysgraphia/PD (1).jpg")
result = predict_handwriting_cnn(image_file)

print("Predicted Label:", result["predicted_label"])
print("Class Probabilities:", result["class_probabilities"])
print("Model:", result.get("model_type", "ResNet-18"))
print("Checkpoint:", result.get("checkpoint", "default"))
```

Both return: `predicted_label`, `class_probabilities`, `model_metadata`, and a non-diagnostic disclaimer.

---

## Dataset Attribution

- **Source**: Mendeley Data, *Potential Dysgraphia Handwriting Dataset of School-Age Children*, Ramlan et al. (Oct 2023), DOI: [10.17632/39hr8dx76p.1](https://data.mendeley.com/datasets/39hr8dx76p/1)
- **License**: Follow the provider's terms and attribution requirements
- **Placement**: Class folders under `data\raw\` as shown above

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: app.*` | Ensure repo root is on `sys.path` (handled by `app/main.py`) |
| `CNN model not available` | Train CNN first: `python -m src.models.cnn_train ...` |
| `Feature dataset not found` | Run `python -m src.features.build_feature_dataset` |
| `Port 8501 in use` | Use `streamlit run app/main.py --server.port 8502` |
| PowerShell script execution blocked | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| `raw_image1.jpeg` missing | File is in `data/raw/` — do not delete |
| Tests fail with dataset errors | Expected — tests requiring dataset are skipped when unavailable |

---

## Testing

```powershell
# Run full test suite
python -m pytest tests/ -q

# Run with coverage
python -m pytest tests/ --cov=src --cov=app -q

# Run specific test module
python -m pytest tests/test_inference.py -v
```

- **51 tests** (unit + integration)
- Tests requiring dataset/artifacts are **skipped** when unavailable
- Synthetic preprocessing, validation, schema, and model tests run without dataset

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes with clear, focused commits
4. Ensure tests pass: `python -m pytest -q`
5. Check style: `git diff --check`
6. Open a Pull Request with description of changes

### Code Quality Standards
- Modular architecture: `app/main.py` routes; components in `app/components/`; services in `app/services/`
- No circular imports; shared charts in `app/components/charts.py`
- No dummy/fake data — handle missing artifacts gracefully
- Unique, stable Plotly keys for all charts
- Preserve existing inference logic; do not retrain models in PRs

---

## License & Disclaimer

This project is an **academic research prototype** for screening support only. It does not provide medical, clinical, psychiatric, or educational diagnoses. Raw handwriting images and generated outputs are excluded from Git by `.gitignore`.

**Dataset**: Mendeley Data (DOI: 10.17632/39hr8dx76p.1) — follow provider's license.