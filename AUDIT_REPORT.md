# GraphiSense — Repository Audit and Streamlit UI Modernization Plan

**Date:** 2026-10-26  
**Repository Root:** `C:\Users\samri\graphisense`  
**Git Status:** Clean, on `main`, up to date with `origin/main`

---

## 1. Executive Summary

GraphiSense is a **well-structured, tested, and documented** academic research prototype for static handwriting analysis targeting potential dysgraphia screening support. The project implements a complete end-to-end pipeline:

- **Image preprocessing** (loading, grayscale conversion, polarity-aware Otsu/Triangle binarization)
- **25 static handwriting features** (geometry, ink density, connected components, spatial distribution, stroke width proxy)
- **Random Forest classifier** trained via stratified 5-fold CV on 9 reduced static features
- **Deterministic inference engine** loading frozen `model_pipeline.joblib` + metadata
- **Streamlit web application** with upload/sample selection, preprocessing visualization, feature tables, probability display, and model metadata
- **Comprehensive test suite** (42 tests passing, 17 skipped when dataset/model artifacts unavailable)
- **EDA and evaluation reports** with visual artifacts

**Key Finding:** The repository is **production-quality for its stated scope** (Random Forest baseline). No CNN/ResNet implementation exists. The current UI is functional but minimal; significant UX improvements can be made within Streamlit.

---

## 2. Repository File Inventory

| Category | Files | Status |
|----------|-------|--------|
| **Entry Points** | `app/main.py`, `app/pipeline_handler.py` | ✅ Implemented |
| **Preprocessing** | `src/preprocessing/image_loader.py`, `binarization.py`, `__init__.py` | ✅ Implemented & tested |
| **Feature Extraction** | `src/features/handwriting_features.py`, `build_feature_dataset.py`, `analyze_features.py`, `__init__.py` | ✅ Implemented & tested |
| **ML Models** | `src/models/train_evaluate.py`, `inference.py`, `__init__.py` | ✅ Implemented & tested |
| **Visualization** | `src/visualization/demo_inference.py`, `__init__.py` | ✅ Implemented |
| **Tests** | `tests/test_*.py` (7 files) | ✅ 42 passed, 17 skipped |
| **Notebooks** | `notebooks/Stepwise-Guided-Notebook.ipynb`, `inference_demo.ipynb` | ✅ Present (empty scaffold) |
| **Artifacts** | `outputs/models/`, `outputs/figures/`, `outputs/features/`, `outputs/reports/` | ✅ Present |
| **Config** | `requirements.txt`, `README.md` | ✅ Complete |

---

## 3. Existing Pipeline and Architecture

### End-to-End Flow (Verified by Code Inspection)

```
Image Upload / Sample Selection
        │
        ▼
load_and_preprocess_image() [pipeline_handler.py]
  ├─ decode_image_bytes() → cv2.imdecode
  ├─ load_image() → cv2.imread + fallback
  ├─ to_grayscale() → cv2.cvtColor (BGR2GRAY/BGRA2GRAY)
  ├─ detect_background_is_light() → border sampling (5% margins)
  └─ binarize_handwriting() → Otsu/Triangle/Fixed + polarity handling
        │
        ▼
extract_handwriting_features() [handwriting_features.py]
  ├─ 25 static features extracted:
  │   • Image geometry: width, height, aspect_ratio
  │   • Ink density, foreground_pixels
  │   • Connected components: count, width/height/area stats, aspect ratios
  │   • Spatial: centroid (raw + normalized), row projections, active_row_ratio
  │   • Stroke width proxy: distance transform medial axis (2×R)
  │
  └─ min_component_area=5 filters noise speckles
        │
        ▼
predict_handwriting() [inference.py]
  ├─ load_saved_model() → joblib.load(model_pipeline.joblib) + JSON metadata
  ├─ Filter/order features per metadata['feature_names'] (9 reduced features)
  ├─ pipeline.predict() + pipeline.predict_proba()
  └─ Return structured dict with label, probabilities, features, metadata, disclaimer
        │
        ▼
Streamlit UI [main.py]
  ├─ Header + Input column (sample/upload + polarity control)
  ├─ "Analyze handwriting" button → spinner + inference
  ├─ Results: Prediction card, probability bars, side-by-side images
  └─ Tabs: Feature Analysis, Image Processing, Model Details, About & Limitations
```

### Verified Artifacts (Exist & Load)

| Artifact | Path | Verified |
|----------|------|----------|
| Trained Pipeline | `outputs/models/model_pipeline.joblib` | ✅ 313KB, loads via joblib |
| Model Metadata | `outputs/models/model_metadata.json` | ✅ 9 features, RF, 249 samples |
| Feature Dataset | `outputs/features/features.csv` | ✅ 249 rows, 25 features, 0 missing |
| Evaluation Metrics | `outputs/reports/model_evaluation_metrics.csv` | ✅ Present |
| Evaluation Report | `outputs/reports/model_evaluation_report.md` | ✅ Present |
| EDA Report | `outputs/reports/eda_report.md` | ✅ Present |
| Figures | `outputs/figures/*.png` (11 files) | ✅ All present & non-empty |

---

## 4. Random Forest Implementation Status

| Aspect | Status | Evidence |
|--------|--------|----------|
| **Model Architecture** | ✅ Implemented & Verified | `train_evaluate.py:175-187` — `RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=4)` |
| **Feature Set** | ✅ Implemented & Verified | `REDUCED_STATIC_FEATURES` (9 features) — `train_evaluate.py:50-60` |
| **Training Pipeline** | ✅ Implemented & Verified | `train_and_save_final_model()` — `train_evaluate.py:537-638` |
| **Cross-Validation** | ✅ Implemented & Verified | `evaluate_model_cv()` — stratified 5-fold, zero leakage via `Pipeline` — `train_evaluate.py:194-286` |
| **Model Persistence** | ✅ Implemented & Verified | `joblib.dump(pipeline, model_path)` + JSON metadata — `train_evaluate.py:587-636` |
| **Inference Engine** | ✅ Implemented & Verified | `predict_handwriting()` — loads frozen model, no refit — `inference.py:124-268` |
| **Feature Importance** | ✅ Implemented & Verified | `global_feature_importance_or_coef` in metadata — `model_metadata.json:36-46` |
| **Evaluation Metrics** | ✅ Implemented & Verified | Accuracy, Balanced Accuracy, Precision, Recall (PD/LPD), F1, ROC-AUC — `train_evaluate.py:243-276` |
| **Confusion Matrix** | ✅ Implemented & Verified | OOF cumulative CM saved as figure — `train_evaluate.py:356-386` |
| **ROC Curves** | ✅ Implemented & Verified | OOF ROC curves for all 4 configs — `train_evaluate.py:389-419` |
| **Test Coverage** | ✅ Implemented & Verified | `tests/test_models.py`, `test_inference.py` — 15 tests for models/inference |

**Summary:** Random Forest mode is **fully implemented, tested, and verified**. The model artifact exists and loads correctly.

---

## 5. CNN/ResNet Implementation Status

| Aspect | Status | Evidence |
|--------|--------|----------|
| **CNN Architecture** | ❌ Missing | No PyTorch/TensorFlow/Keras imports anywhere. `requirements.txt` has only sklearn. |
| **ResNet / Pretrained Integration** | ❌ Missing | No `torchvision`, `timm`, or similar. |
| **Dataset Loader for DL** | ❌ Missing | Current loader returns NumPy arrays for sklearn, not DataLoaders. |
| **Saved Checkpoints** | ❌ Missing | Only `model_pipeline.joblib` (sklearn RF) exists. |
| **Training/Validation History** | ❌ Missing | No training curves, no DL logging. |
| **Inference Pipeline** | ❌ Missing | `inference.py` only supports sklearn pipelines. |
| **Grad-CAM / Explainability** | ❌ Missing | No deep learning framework present. |
| **Evaluation Metrics** | ❌ Missing | No DL evaluation code. |

**Conclusion:** **Mode B (Advanced CNN/ResNet) does not exist at all.** The entire codebase is pure OpenCV + scikit-learn. Implementing Mode B requires adding a deep learning framework (recommend PyTorch + torchvision for ResNet-18 transfer learning), a dataset wrapper returning tensors, training loop, checkpointing, and a new inference path.

---

## 6. Streamlit UI Assessment

### Current Structure (`app/main.py`)

| Component | Implementation | Quality |
|-----------|---------------|---------|
| **Landing Page** | Single-page with custom CSS header | Minimal but clean |
| **Navigation** | None (single page, tabs for details) | Basic |
| **Input Selection** | Radio: Sample Image / Upload Image | Functional |
| **Sample Images** | 4 hardcoded dataset samples | Limited |
| **Upload Widget** | `st.file_uploader` (JPG/PNG) | Standard |
| **Polarity Control** | Selectbox: auto/dark/light | Good UX |
| **Preprocessing Preview** | Live preview on input change | ✅ Good |
| **Analyze Button** | Primary button with spinner | Standard |
| **Results Display** | Prediction card + probability bars | Functional |
| **Image Comparison** | Side-by-side original + binary mask | ✅ Core visualization |
| **Feature Table** | `st.dataframe` in "Feature Analysis" tab | Basic table |
| **All 25 Features** | Expander with full feature list | Available |
| **Distance Transform** | Colored heatmap in "Image Processing" tab | ✅ Good technical viz |
| **Model Metadata** | 3 metric cards + JSON expander | Basic |
| **Disclaimer Tab** | Static markdown | Complete |
| **Footer** | Persistent disclaimer | ✅ |

### Why the Current UI Feels Basic

1. **No Analysis Mode Selection** — Only Random Forest exists; no toggle for "Basic" vs "Advanced"
2. **Static Layout** — Fixed two-column; no responsive adaptation
3. **Limited Visualizations** — Only tables and basic probability bars; no interactive charts (Plotly), no confusion matrix, no feature importance chart, no ROC curve
4. **No Loading States Beyond Spinner** — No progress breakdown (preprocessing → features → inference)
5. **No Error Recovery UI** — Errors shown as red boxes; no guidance
6. **No Image Segmentation Overlay** — Binary mask only; no bounding boxes, line segmentation visualization
7. **Typography/Spacing Inconsistent** — Custom CSS but not systematic
8. **No Model Availability Indicator** — User doesn't know if model is loaded until they click Analyze
9. **Sample Selection Hardcoded** — Only 4 samples; no gallery view
10. **No Export/Report Feature** — Cannot download results

---

## 7. Visualization Opportunities (Existing Data Available)

| Chart | Data Source | Priority | Library |
|-------|-------------|----------|---------|
| Feature Importance (horizontal bar) | `model_metadata.json["global_feature_importance_or_coef"]` | P1 | Plotly |
| Confusion Matrix Heatmap | `train_evaluate.py` OOF predictions → regenerate or load from saved CSV | P1 | Plotly/Matplotlib |
| ROC Curve | `outputs/figures/roc_curves.png` exists; can regenerate from OOF probs | P1 | Plotly |
| Feature Distribution by Class (violin/box) | `features.csv` — 249 rows × 25 features | P1 | Plotly |
| Training vs Validation Loss/Acc | ❌ Not available (sklearn CV, no epoch history) | N/A | — |
| Grad-CAM | ❌ Requires CNN model | P2 (if Mode B added) | — |
| Stroke Width Distribution (per image) | `cv2.distanceTransform` ridges per inference | P1 | Plotly |
| Component Size Distribution | Connected component stats per inference | P1 | Plotly |
| Original vs Processed vs Distance Transform (3-panel) | Already generated in `demo_inference.py` | P0 | Matplotlib/Streamlit |

**Dependencies to Add for Visualizations:**
- `plotly` — for interactive charts (feature importance, confusion matrix, distributions)
- Optional: `kaleido` for static export

---

## 8. Prioritized File-by-File Change Plan

### P0: Correctness & Broken UI Issues

| File | Change | Rationale |
|------|--------|-----------|
| `app/main.py` | Add model availability check at startup; show status badge | Prevents "Analyze" click → FileNotFoundError |
| `app/main.py` | Wrap inference in try/except with user-friendly error messages | Current raw exception display is poor UX |
| `app/main.py` | Validate image_bytes size before processing | Prevents OOM on huge uploads |
| `app/pipeline_handler.py` | Ensure `run_app_inference` returns consistent dict on error | Current raises exceptions caught by UI |

### P1: Mode Selection, Routing & Core Visualizations

| File | Change | Rationale |
|------|--------|-----------|
| `app/main.py` | Add **Analysis Mode** radio: "Basic (Random Forest)" / "Advanced (CNN/ResNet)" | Core requirement |
| `app/main.py` | Disable Advanced mode with tooltip "Model not trained — coming in v2" | Honest about availability |
| `app/main.py` | Refactor layout: Landing → Mode Select → Upload → Process → Results tabs | Professional flow |
| `app/main.py` | Add Plotly feature importance chart in "Feature Analysis" tab | Uses existing metadata |
| `app/main.py` | Add Plotly confusion matrix heatmap in "Model Details" tab | Regenerate from OOF or load saved |
| `app/main.py` | Add interactive feature distribution chart (violin/box by class) | Uses `features.csv` |
| `app/main.py` | Add stroke width distribution histogram (per-image ridges) | Computed live in `Image Processing` tab |
| `app/main.py` | Replace static probability bars with Plotly gauge/bar | Better visual |
| `app/main.py` | Add expandable "Processing Details" showing timings per stage | Transparency |

### P1: Layout, Navigation & States

| File | Change | Rationale |
|------|--------|-----------|
| `app/main.py` | Migrate to `st.tabs` or `st.navigation` (Streamlit 1.45+) for top-level pages | Modern navigation |
| `app/main.py` | Use `st.columns` with `gap="large"` consistently; add responsive breakpoints | Layout consistency |
| `app/main.py` | Add loading skeleton placeholders while inference runs | Perceived performance |
| `app/main.py` | Standardize card styling (prediction, metadata, feature rows) | Visual consistency |
| `app/main.py` | Add empty state illustrations for each tab | Polish |

### P2: CNN/ResNet Integration (If Mode B Implemented)

| File | Change | Rationale |
|------|--------|-----------|
| `requirements.txt` | Add `torch`, `torchvision`, `timm` (or `torchvision.models.resnet18`) | DL framework |
| `src/models/` | New `cnn_train.py`, `cnn_inference.py` | Separate DL pipeline |
| `src/models/__init__.py` | Export CNN functions | API consistency |
| `app/pipeline_handler.py` | Add `run_cnn_inference()` routing | Mode separation |
| `app/main.py` | Enable Advanced mode when CNN checkpoint exists | Feature flag |
| `src/visualization/` | Add `gradcam.py` for Grad-CAM | Explainability |
| `notebooks/` | Add CNN training demo notebook | Documentation |

### P2: Optional Enhancements

| File | Change | Rationale |
|------|--------|-----------|
| `app/main.py` | Add "Download Report" button (PDF/Markdown) | Sharing results |
| `app/main.py` | Add session history (last N analyses) | Usability |
| `app/main.py` | Add keyboard shortcuts (Enter to analyze) | Power users |
| `src/features/` | Add SHAP explainer for RF (sample-level) | Deeper interpretability |

---

## 9. Dependencies to Add

| Package | Version | Justification | Priority |
|---------|---------|---------------|----------|
| `plotly` | ≥5.18 | Interactive charts: feature importance, confusion matrix, distributions, probability gauges | **P1** |
| `kaleido` | ≥0.2 | Static export of Plotly figures for report download | P2 |
| `torch` | ≥2.0 | CNN/ResNet training & inference (Mode B) | **P2** |
| `torchvision` | ≥0.15 | ResNet-18 pretrained weights, transforms | **P2** |
| `timm` | ≥0.9 | Alternative model zoo (optional, if more architectures needed) | P2 |

**No changes to existing dependencies.** All current packages (opencv-python, numpy, pandas, matplotlib, scikit-learn, scipy, joblib, streamlit, pytest) are correctly versioned and used.

---

## 10. Testing Plan & Acceptance Criteria

### Current Test Status: ✅ 42 passed, 17 skipped (dataset/model dependent)

### Additional Tests Needed

| Test | Location | Acceptance Criteria |
|------|----------|---------------------|
| UI: Mode selection renders correctly | `tests/test_app.py` | Radio shows Basic/Advanced; Advanced disabled with tooltip |
| UI: Model availability badge | `tests/test_app.py` | Green badge when `model_pipeline.joblib` exists |
| UI: Feature importance chart renders | `tests/test_app.py` | Plotly chart appears in Feature Analysis tab with 9 bars |
| UI: Confusion matrix heatmap renders | `tests/test_app.py` | Plotly heatmap in Model Details tab |
| UI: Probability gauge renders | `tests/test_app.py` | Interactive Plotly gauge replaces static bars |
| UI: Error handling — missing model | `tests/test_app.py` | Friendly error message, not traceback |
| UI: Error handling — corrupt upload | `tests/test_app.py` | Clear "Invalid image file" message |
| Integration: End-to-end Basic mode | `tests/test_app.py` | Sample → Analyze → Results displayed without error |
| Integration: CNN inference (when implemented) | New `tests/test_cnn.py` | Checkpoint loads, prediction returns probabilities |

### Manual Acceptance Criteria (UI)

1. **Landing Page** — Branding, description, mode selector, upload area all visible without scroll on 1920×1080
2. **Basic Mode** — Upload → Analyze → Results in <3s (CPU); all tabs populated
3. **Advanced Mode** — Disabled with clear "Coming Soon" messaging
4. **Visualizations** — All charts interactive (hover, zoom); exportable via camera icon
5. **Responsive** — Layout adapts to 768px width (tablet) without horizontal scroll
6. **Accessibility** — Color contrast ≥4.5:1; keyboard navigable; alt text on images

---

## 11. Risks, Blockers & Unresolved Questions

### Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| CNN training requires GPU | High | Delays Mode B | Document CPU fallback; use smaller ResNet-18; pretrained weights reduce epochs |
| Dataset has no writer IDs → leakage | Known | Overestimated CV metrics | Already documented in README & reports; no fix possible without new data |
| Stroke width in pixels not DPI-normalized | Known | Limits generalization | Already flagged in metadata; UI should show resolution disclaimer |
| Plotly adds ~15MB to deploy size | Medium | Larger container | Acceptable for desktop; optional for cloud |

### Blockers

1. **No CNN code exists** — Mode B implementation is net-new work (est. 2-3 weeks for ResNet-18 transfer learning + training + integration)
2. **No writer-level splits possible** — Fundamental dataset limitation; affects any DL model's claimed generalization

### Unresolved Questions

1. **Target Deployment** — Local desktop only? Streamlit Cloud? Docker? Affects dependency strategy (Plotly/kaleido size).
2. **Mode B Scope** — Binary PD/LPD only, or multi-class? Current dataset is binary.
3. **Explainability Requirement** — Is Grad-CAM mandatory for Mode B, or is feature importance sufficient?
4. **Report Export Format** — PDF (needs `weasyprint`/`reportlab`), Markdown, or HTML?
5. **Real-time Constraints** — Is <1s inference required? Current RF is ~200ms; ResNet-18 ~50ms on CPU.

---

## 12. Recommended Next Steps (Implementation Order)

1. **Week 1 (P0+P1 UI):** Implement mode selector, model availability badge, Plotly feature importance, confusion matrix, probability gauge, responsive layout fixes, error handling polish. All within `app/main.py` + `requirements.txt` (add plotly).
2. **Week 2 (P1 Visualizations):** Add feature distribution charts (violin/box by class), stroke width histogram, component size distribution, processing timeline. Connect to existing `features.csv` and live inference data.
3. **Week 3 (P2 CNN — Optional):** Add PyTorch, implement ResNet-18 transfer learning in `src/models/cnn_train.py`, create dataset wrapper, training script, checkpoint saving. Integrate inference path in `pipeline_handler.py`.
4. **Week 4 (P2 Polish):** Grad-CAM visualization, report export (Markdown/PDF), session history, final testing.

---

**Audit Complete.** No files modified during audit. All findings reference actual code locations. The repository is ready for the proposed minimal-change modernization.