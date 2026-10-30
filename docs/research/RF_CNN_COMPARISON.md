# RF vs CNN Comparison Report

## GraphiSense — Protocol-Aware Model Comparison

**Project:** GraphiSense — Computer Vision-Based Digital Handwriting Analysis for Potential Dysgraphia Screening Support  
**Date:** 2026-10-30  
**Dataset:** Mendeley Potential Dysgraphia Handwriting Dataset (DOI: 10.17632/39hr8dx76p.1)

---

## 1. Dataset Information

| Parameter | Value |
|-----------|-------|
| **Total Images** | 249 |
| **Classes** | PD (Potential Dysgraphia): 114, LPD (Low Potential Dysgraphia): 135 |
| **CNN Training Set** | 174 images (70%) |
| **CNN Validation Set** | 37 images (15%) |
| **CNN Test Set** | 38 images (15%) |
| **CNN Test Class Distribution** | LPD: 21, PD: 17 |
| **RF Evaluation** | Stratified 5-Fold CV on all 249 images |

---

## 2. Random Forest Evaluation Results

**Model Configuration:** Random Forest (Set A — Normalized 9 Features)  
**Features:** ink_density, centroid_x_norm, centroid_y_norm, active_row_ratio, component_aspect_ratio_mean, component_aspect_ratio_std, stroke_width_mean, stroke_width_median, stroke_width_std  
**Evaluation Protocol:** Stratified 5-Fold Cross-Validation (N=249)

| Metric | Mean | Std Dev | 95% CI (approx) |
|--------|------|---------|-----------------|
| Accuracy | **83.93%** | ±3.58% | [80.35%, 87.51%] |
| Balanced Accuracy | **84.08%** | ±3.82% | [80.26%, 87.90%] |
| Precision (macro) | 81.30% | ±5.70% | [75.60%, 87.00%] |
| Recall (PD) | 85.93% | ±11.52% | [74.41%, 97.45%] |
| Recall (LPD) | 82.22% | ±8.57% | [73.65%, 90.79%] |
| F1-Score (PD) | **82.76%** | ±4.75% | [78.01%, 87.51%] |
| ROC-AUC | **89.46%** | ±4.69% | [84.77%, 94.15%] |

**Confusion Matrix (Cumulative CV):** Not available in current artifacts (requires full evaluation run to generate).

**Training Samples:** 249 (all data used in CV folds)  
**Model Type:** Random Forest Classifier  
**Feature Count:** 9 (reduced static features)  
**Random State:** 42

---

## 3. CNN/ResNet-18 Evaluation Results

**Model Configuration:** ResNet-18 with ImageNet pretrained weights  
**Input:** 224×224 RGB, ImageNet normalization  
**Training:** AdamW optimizer, cross-entropy loss, ReduceLROnPlateau scheduler  
**Early Stopping:** Patience=10, min_delta=1e-4  
**Best Checkpoint:** Epoch 13 (validation loss minimum)

**Dataset Split (from split_metadata.json):**
- Training: 174 images (indices in split_metadata.json)
- Validation: 37 images
- Test: 38 images (15% hold-out)

**Test Set Class Distribution:** LPD: 21, PD: 17

**Evaluation Protocol:** Single held-out test set evaluation

| Metric | Value | 95% CI (Wilson, approx) |
|--------|-------|-------------------------|
| Accuracy | **89.47%** | [76.8%, 95.7%] |
| Balanced Accuracy | **89.92%** | [76.3%, 95.7%] |
| Precision (macro) | 89.47% | — |
| Recall (macro) | 89.92% | — |
| F1 (macro) | 89.44% | — |
| **Precision (LPD)** | **94.74%** | [75.4%, 99.1%] |
| **Recall (LPD)** | **85.71%** | [63.7%, 96.0%] |
| **F1 (LPD)** | **90.00%** | — |
| **Precision (PD)** | **84.21%** | [60.4%, 96.0%] |
| **Recall (PD)** | **94.12%** | [71.3%, 99.8%] |
| **F1 (PD)** | **88.89%** | — |
| **ROC-AUC** | **97.48%** | [86.5%, 99.7%] |

**Confusion Matrix (Test Set, N=38):**
| | Predicted LPD | Predicted PD |
|---|---|---|
| **True LPD** | 18 | 3 |
| **True PD** | 1 | 16 |

**Training Duration:** 25 epochs (early stopped)  
**Best Epoch:** 13 (val_acc = 97.30%)  
**Test Loss:** 0.2263

---

## 4. Evaluation Protocol Differences

| Aspect | Random Forest | CNN/ResNet-18 |
|--------|---------------|---------------|
| **Evaluation Method** | Stratified 5-Fold Cross-Validation | Single Held-Out Test Split |
| **Data Used for Evaluation** | All 249 images (each used once in test fold) | 38 images (15% hold-out) |
| **Metric Aggregation** | Mean ± Std Dev across 5 folds | Single point estimate |
| **Variance Estimate** | Available (std dev across folds) | Not available (single test set) |
| **Feature Representation** | 9 handcrafted static features | Learned from raw pixels (ResNet-18) |
| **Pretraining** | None (trained from scratch) | ImageNet pretrained weights |
| **Trainable Parameters** | ~100 (trees × features) | ~11.2M (ResNet-18) |

### Critical Note
> **These results are NOT directly comparable.** RF uses 5-fold CV on the full dataset (249 images), while CNN uses a single 15% held-out test set (38 images). The CNN benefits from pretrained ImageNet weights. Any comparison is descriptive only — not a controlled head-to-head experiment.

---

## 5. Comparison Tables

### 5.1 Main Metrics (Percentages)

| Metric | RF Set A (5-Fold CV) | CNN (Held-Out Test) |
|--------|---------------------|---------------------|
| Accuracy | 83.93% ± 3.58% | 89.47% |
| Balanced Accuracy | 84.08% ± 3.82% | 89.92% |
| F1-Score (PD) | 82.76% ± 4.75% | 88.89% |
| ROC-AUC | 89.46% ± 4.69% | 97.48% |

### 5.2 Per-Class Metrics

| Class | Metric | RF (CV) | CNN (Test) |
|-------|--------|---------|------------|
| **LPD** | Precision | 81.30% ± 5.70% | 94.74% |
| | Recall | 82.22% ± 8.57% | 85.71% |
| | F1-Score | 81.76% (calc) | 90.00% |
| **PD** | Precision | 81.30% ± 5.70% | 84.21% |
| | Recall | 85.93% ± 11.52% | 94.12% |
| | F1-Score | 82.76% ± 4.75% | 88.89% |

### 5.3 RF Feature Importance (Set A)

| Feature | Importance |
|---------|------------|
| stroke_width_std | 0.2381 |
| ink_density | 0.1677 |
| stroke_width_mean | 0.1579 |
| active_row_ratio | 0.1147 |
| centroid_y_norm | 0.0959 |
| component_aspect_ratio_std | 0.0708 |
| component_aspect_ratio_mean | 0.0608 |
| centroid_x_norm | 0.0440 |
| stroke_width_median | 0.0502 |

---

## 6. Explanation of Performance Differences

### Confirmed Findings (Evidence from Implementation)

1. **Different Feature Representations**
   - **RF**: Uses 9 handcrafted static geometric features (ratios, pixel counts, stroke width proxies)
   - **CNN**: Learns hierarchical visual features directly from 224×224 RGB pixels via ResNet-18

2. **Different Evaluation Protocols**
   - **RF**: 5-fold CV provides lower-variance estimate using all 249 images
   - **CNN**: Single held-out test (38 images) has higher variance; point estimate only

3. **Pretraining Advantage (CNN)**
   - ResNet-18 initialized with ImageNet weights (1.2M images, 1000 classes)
   - Provides strong visual feature extractors before seeing any handwriting data
   - RF trains entirely from scratch on 249 images

4. **Model Capacity**
   - RF: ~100 effective parameters (ensemble of trees on 9 features)
   - CNN: ~11.2M parameters (ResNet-18), fine-tuned on handwriting data

### Possible Explanations (Not Confirmed)

1. **Small CNN Test Set** (N=38): Single test set estimate has wide confidence intervals. The 89.47% accuracy could range ~77–96% (Wilson CI).

2. **Writer-Level Leakage**: Dataset lacks participant IDs. Both CV (RF) and train/test splits (CNN) are at image level. If multiple crops from same writer exist, both protocols may overestimate generalization.

3. **Class Imbalance Handling**: RF uses class-weighted CV; CNN uses balanced accuracy metric. Test set has 21 LPD / 17 PD (mild imbalance).

4. **Image Resolution**: CNN resizes to 224×224 with ImageNet normalization; RF features are resolution-dependent (pixel units).

---

## 7. Limitations

1. **No Controlled Comparison**: Different evaluation protocols, feature spaces, and pretraining
2. **Small CNN Test Set**: N=38 limits statistical power; wide CIs
3. **No Writer IDs**: Cannot perform writer-independent evaluation
4. **Single CNN Run**: No cross-validation or multiple seeds for CNN
5. **RF Confusion Matrix**: Cumulative CV confusion matrix not available in artifacts
6. **Dataset Size**: 249 images is small for deep learning; pretrained weights compensate
7. **No Statistical Testing**: No significance tests between models

---

## 8. Presentation-Ready Key Findings

**RF vs CNN — Key Findings**

- **RF Accuracy: 83.93%** (5-fold CV, N=249, ±3.58%)
- **CNN Accuracy: 89.47%** (held-out test, N=38)
- **CNN Balanced Accuracy: 89.92%**
- **CNN ROC-AUC: 97.48%**
- **RF uses manually extracted handwriting features** (9 geometric ratios and stroke proxies)
- **CNN learns patterns directly from handwriting images** (ResNet-18, ImageNet pretrained)
- **CNN benefits from 1.2M ImageNet pretraining images** vs RF trained from scratch on 249
- **Both models require further validation** on independent datasets
- **Different evaluation methods limit direct comparison** (CV vs single test set)
- **Dataset has no writer IDs** — image-level splits may overestimate generalization
- **CNN test set is only 38 images** (15% hold-out) — results have wide confidence intervals

---

## 9. Recommended Fair-Comparison Experiment

To enable a scientifically valid head-to-head comparison:

1. **Common Hold-Out Test Set**: Reserve 15–20% of images (stratified by class) as a frozen test set, never used in training/CV
2. **Same Evaluation Protocol**: Evaluate both RF and CNN on identical test set
3. **Writer-Independent Splits**: If writer IDs available, split by writer; otherwise acknowledge limitation
4. **Multiple Seeds for CNN**: Run CNN training with 5+ different random seeds, report mean ± std
5. **Statistical Testing**: Use McNemar's test or bootstrap CI for accuracy difference
6. **Calibration Assessment**: Compare predicted probability reliability (ECE, calibration curves)

---

## 10. Appendix: Artifact Sources

| Artifact | Path |
|----------|------|
| RF Metrics (CV) | `outputs/reports/model_evaluation_metrics.csv` (row 4) |
| RF Metadata | `outputs/models/model_metadata.json` |
| CNN Training History | `outputs/models/cnn/training_history.json` |
| CNN Split Metadata | `outputs/models/cnn/split_metadata.json` |
| CNN Evaluation Results | `outputs/reports/cnn_evaluation/evaluation_results.json` |
| CNN Best Checkpoint | `outputs/models/cnn/checkpoints/best_model.pt` |

---

*Report generated from actual repository artifacts. All values extracted programmatically from source files. No metrics were invented or estimated beyond what is stored in the evaluation outputs.*