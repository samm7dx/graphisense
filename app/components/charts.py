"""Charts component for reusable Plotly charts."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd


def plot_feature_importance(importance_dict: dict) -> go.Figure:
    """Create horizontal bar chart for feature importance."""
    df = pd.DataFrame(
        sorted(importance_dict.items(), key=lambda x: x[1], reverse=True),
        columns=["Feature", "Importance"]
    )
    fig = go.Figure(go.Bar(
        x=df["Importance"],
        y=df["Feature"],
        orientation="h",
        marker_color="#0284C7",
        text=[f"{v:.3f}" for v in df["Importance"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title="Global Feature Importance (Random Forest)",
        xaxis_title="Mean Decrease in Impurity",
        yaxis_title="",
        height=350,
        margin=dict(l=160, r=40, t=60, b=40),
        plot_bgcolor="white",
        font=dict(family="Inter, system-ui", size=12),
    )
    fig.update_xaxes(gridcolor="#E2E8F0", zerolinecolor="#E2E8F0")
    fig.update_yaxes(gridcolor="#E2E8F0", autorange="reversed")
    return fig


def plot_confusion_matrix(cm: np.ndarray, labels: list = ["LPD", "PD"]) -> go.Figure:
    """Create confusion matrix heatmap."""
    cm = np.array(cm)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    text = [[f"{cm[i, j]}<br>({cm_norm[i, j]:.1%})" for j in range(2)] for i in range(2)]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[f"Predicted {l}" for l in labels],
        y=[f"True {l}" for l in labels],
        colorscale="Blues",
        showscale=True,
        text=text,
        texttemplate="%{text}",
        textfont={"size": 14},
        hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title="Out-of-Fold Confusion Matrix (5-Fold CV)",
        height=350,
        margin=dict(l=100, r=40, t=60, b=80),
        font=dict(family="Inter, system-ui", size=12),
    )
    return fig


def plot_feature_distributions_grouped(
    features_df: pd.DataFrame,
    feature_names: list,
    current_features: dict = None,
) -> list[go.Figure]:
    """Create grouped violin plots by feature scale/category.

    Groups features with comparable ranges:
    - Group 1: Ink density & normalized geometry [0-1]
    - Group 2: Component aspect ratios [~0.5-3]
    - Group 3: Stroke width statistics [pixels]

    Returns list of figures, one per group.
    """
    from plotly.subplots import make_subplots

    colors = {"PD": "#F59E0B", "LPD": "#34D399"}

    # Define feature groups by semantic category and scale
    feature_groups = {
        "Ink Density & Normalized Geometry": [
            "ink_density",
            "centroid_x_norm",
            "centroid_y_norm",
            "active_row_ratio",
        ],
        "Component Aspect Ratios": [
            "component_aspect_ratio_mean",
            "component_aspect_ratio_std",
        ],
        "Stroke Width Statistics (px)": [
            "stroke_width_mean",
            "stroke_width_median",
            "stroke_width_std",
        ],
    }

    figures = []

    for group_name, group_features in feature_groups.items():
        # Filter to only features that exist in the dataframe and are requested
        valid_features = [f for f in group_features if f in features_df.columns and f in feature_names]
        if not valid_features:
            continue

        n = len(valid_features)
        cols = min(2, n)
        rows = (n + cols - 1) // cols

        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f.replace("_", " ").title() for f in valid_features],
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
        )

        for idx, feat in enumerate(valid_features):
            row = idx // cols + 1
            col = idx % cols + 1

            for cls in ["PD", "LPD"]:
                cls_data = features_df[features_df["class_label"] == cls][feat].dropna()
                if len(cls_data) == 0:
                    continue

                fig.add_trace(go.Violin(
                    y=cls_data,
                    name=f"{cls}",
                    legendgroup=cls,
                    scalegroup=feat,
                    side="positive" if cls == "PD" else "negative",
                    line_color=colors[cls],
                    fillcolor=colors[cls],
                    opacity=0.6,
                    meanline_visible=True,
                    box_visible=True,
                    width=0.8,
                    showlegend=(idx == 0),
                    hovertemplate=f"<b>{feat.replace('_', ' ').title()}</b><br>Class: {cls}<br>Value: %{{y:.4f}}<extra></extra>",
                ), row=row, col=col)

            # Add current sample marker if provided
            if current_features and feat in current_features:
                val = current_features[feat]
                fig.add_trace(go.Scatter(
                    x=[0.5], y=[val],
                    mode="markers",
                    marker=dict(size=12, color="#DC2626", symbol="diamond", line=dict(width=2, color="white")),
                    name="Current Sample",
                    showlegend=(idx == 0),
                    hovertemplate=f"<b>{feat.replace('_', ' ').title()}</b><br>Current: {val:.4f}<extra></extra>",
                ), row=row, col=col)

        fig.update_layout(
            title=group_name,
            height=250 * rows + 80,
            margin=dict(l=60, r=40, t=80, b=60),
            plot_bgcolor="white",
            font=dict(family="Inter, system-ui", size=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            violinmode="group",
        )
        fig.update_xaxes(showticklabels=False, gridcolor="#E2E8F0")
        fig.update_yaxes(gridcolor="#E2E8F0", title_text="Value")

        figures.append(fig)

    return figures


def plot_feature_distributions(
    features_df: pd.DataFrame,
    feature_names: list,
    current_features: dict = None,
) -> go.Figure:
    """Create violin/box plots for feature distributions by class."""
    fig = go.Figure()

    colors = {"PD": "#F59E0B", "LPD": "#34D399"}

    for i, feat in enumerate(feature_names):
        if feat not in features_df.columns:
            continue
        for cls in ["PD", "LPD"]:
            cls_data = features_df[features_df["class_label"] == cls][feat].dropna()
            if len(cls_data) == 0:
                continue
            fig.add_trace(go.Violin(
                y=cls_data,
                name=f"{cls}",
                legendgroup=cls,
                scalegroup=feat,
                side="positive" if cls == "PD" else "negative",
                line_color=colors[cls],
                fillcolor=colors[cls],
                opacity=0.5,
                meanline_visible=True,
                box_visible=True,
                width=0.8,
                showlegend=(i == 0),
                hovertemplate=f"<b>{feat}</b><br>Class: {cls}<br>Value: %{{y:.4f}}<extra></extra>",
            ))

        # Add current sample marker if provided
        if current_features and feat in current_features:
            val = current_features[feat]
            fig.add_trace(go.Scatter(
                x=[feat], y=[val],
                mode="markers",
                marker=dict(size=14, color="#0F172A", symbol="diamond", line=dict(width=3, color="white")),
                name="Current Sample",
                showlegend=(i == 0),
                hovertemplate=f"<b>{feat}</b><br>Current: {val:.4f}<extra></extra>",
            ))

    fig.update_layout(
        title="Feature Distributions by Class (Violin + Box)",
        xaxis_title="Feature",
        yaxis_title="Value",
        violinmode="group",
        height=500,
        margin=dict(l=60, r=40, t=60, b=100),
        plot_bgcolor="white",
        font=dict(family="Inter, system-ui", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(tickangle=45, gridcolor="#E2E8F0")
    fig.update_yaxes(gridcolor="#E2E8F0")
    return fig


def plot_feature_distributions_separated(
    features_df: pd.DataFrame,
    feature_names: list,
    current_features: dict = None,
) -> go.Figure:
    """Create separated violin plots for each feature - clearer LPD vs PD comparison."""
    from plotly.subplots import make_subplots
    
    colors = {"PD": "#F59E0B", "LPD": "#34D399"}
    n_features = len([f for f in feature_names if f in features_df.columns])
    
    # Create subplots: 3 columns, auto rows
    cols = 3
    rows = (n_features + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f for f in feature_names if f in features_df.columns],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )
    
    feat_idx = 0
    for feat in feature_names:
        if feat not in features_df.columns:
            continue
        
        row = feat_idx // cols + 1
        col = feat_idx % cols + 1
        
        for cls in ["PD", "LPD"]:
            cls_data = features_df[features_df["class_label"] == cls][feat].dropna()
            if len(cls_data) == 0:
                continue
            
            fig.add_trace(go.Violin(
                y=cls_data,
                name=f"{cls}",
                legendgroup=cls,
                scalegroup=feat,
                side="positive" if cls == "PD" else "negative",
                line_color=colors[cls],
                fillcolor=colors[cls],
                opacity=0.6,
                meanline_visible=True,
                box_visible=True,
                width=0.8,
                showlegend=(feat_idx == 0),
                hovertemplate=f"<b>{feat}</b><br>Class: {cls}<br>Value: %{{y:.4f}}<extra></extra>",
            ), row=row, col=col)
        
        # Add current sample marker if provided
        if current_features and feat in current_features:
            val = current_features[feat]
            fig.add_trace(go.Scatter(
                x=[0.5], y=[val],
                mode="markers",
                marker=dict(size=12, color="#DC2626", symbol="diamond", line=dict(width=2, color="white")),
                name="Current Sample",
                showlegend=(feat_idx == 0),
                hovertemplate=f"<b>{feat}</b><br>Current: {val:.4f}<extra></extra>",
            ), row=row, col=col)
        
        feat_idx += 1
    
    fig.update_layout(
        title="Feature Distributions: LPD vs PD (Separated View)",
        height=200 * rows + 100,
        margin=dict(l=60, r=40, t=80, b=60),
        plot_bgcolor="white",
        font=dict(family="Inter, system-ui", size=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        violinmode="group",
    )
    fig.update_xaxes(showticklabels=False, gridcolor="#E2E8F0")
    fig.update_yaxes(gridcolor="#E2E8F0", title_text="Value")
    
    return fig


def plot_class_comparison_radar(
    features_df: pd.DataFrame,
    feature_names: list,
    current_features: dict = None,
) -> go.Figure:
    """Create radar chart comparing mean feature values for LPD vs PD."""
    import numpy as np
    
    colors = {"PD": "#F59E0B", "LPD": "#34D399"}
    
    # Calculate mean values for each class
    pd_means = []
    lpd_means = []
    feat_labels = []
    
    for feat in feature_names:
        if feat not in features_df.columns:
            continue
        pd_data = features_df[features_df["class_label"] == "PD"][feat].dropna()
        lpd_data = features_df[features_df["class_label"] == "LPD"][feat].dropna()
        
        if len(pd_data) > 0 and len(lpd_data) > 0:
            # Normalize to 0-1 range for radar chart
            all_vals = np.concatenate([pd_data, lpd_data])
            min_val, max_val = all_vals.min(), all_vals.max()
            if max_val > min_val:
                pd_mean = (pd_data.mean() - min_val) / (max_val - min_val)
                lpd_mean = (lpd_data.mean() - min_val) / (max_val - min_val)
            else:
                pd_mean = lpd_mean = 0.5
            
            pd_means.append(pd_mean)
            lpd_means.append(lpd_mean)
            feat_labels.append(feat.replace("_", " ").title())
    
    # Close the radar
    pd_means.append(pd_means[0])
    lpd_means.append(lpd_means[0])
    feat_labels.append(feat_labels[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=pd_means,
        theta=feat_labels,
        fill='toself',
        name='Potential Dysgraphia (PD)',
        fillcolor='rgba(245, 158, 11, 0.3)',
        line=dict(color='#F59E0B', width=2),
        hovertemplate="<b>%{theta}</b><br>Normalized Mean: %{r:.2f}<extra></extra>",
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=lpd_means,
        theta=feat_labels,
        fill='toself',
        name='Low Potential Dysgraphia (LPD)',
        fillcolor='rgba(52, 211, 153, 0.3)',
        line=dict(color='#34D399', width=2),
        hovertemplate="<b>%{theta}</b><br>Normalized Mean: %{r:.2f}<extra></extra>",
    ))
    
    if current_features:
        curr_vals = []
        for feat in feature_names:
            if feat not in features_df.columns:
                continue
            if feat in current_features:
                all_vals = np.concatenate([
                    features_df[features_df["class_label"] == "PD"][feat].dropna(),
                    features_df[features_df["class_label"] == "LPD"][feat].dropna()
                ])
                min_val, max_val = all_vals.min(), all_vals.max()
                if max_val > min_val:
                    curr = (current_features[feat] - min_val) / (max_val - min_val)
                else:
                    curr = 0.5
                curr_vals.append(curr)
        curr_vals.append(curr_vals[0])
        
        fig.add_trace(go.Scatterpolar(
            r=curr_vals,
            theta=feat_labels,
            fill='toself',
            name='Current Sample',
            fillcolor='rgba(15, 23, 42, 0.2)',
            line=dict(color='#0F172A', width=2, dash='dot'),
            hovertemplate="<b>%{theta}</b><br>Current: %{r:.2f}<extra></extra>",
        ))
    
    fig.update_layout(
        title="Feature Profile Comparison: LPD vs PD (Normalized Means)",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#E2E8F0"),
            angularaxis=dict(gridcolor="#E2E8F0"),
        ),
        height=500,
        margin=dict(l=80, r=80, t=80, b=80),
        font=dict(family="Inter, system-ui", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
    )
    
    return fig


def plot_probability_gauge(prob_pd: float, prob_lpd: float) -> go.Figure:
    """Create dual gauge chart for class probabilities."""
    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=prob_pd * 100,
        domain={"x": [0, 0.45], "y": [0, 1]},
        title={"text": "Potential Dysgraphia (PD)", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": "#D97706"},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#E2E8F0",
            "steps": [
                {"range": [0, 33], "color": "#FEF3C7"},
                {"range": [33, 66], "color": "#FDE68A"},
                {"range": [66, 100], "color": "#FCD34D"},
            ],
            "threshold": {
                "line": {"color": "#92400E", "width": 3},
                "thickness": 0.8,
                "value": prob_pd * 100,
            },
        },
    ))

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=prob_lpd * 100,
        domain={"x": [0.55, 1], "y": [0, 1]},
        title={"text": "Low Potential Dysgraphia (LPD)", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": "#16A34A"},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#E2E8F0",
            "steps": [
                {"range": [0, 33], "color": "#F0FDF4"},
                {"range": [33, 66], "color": "#DCFCE7"},
                {"range": [66, 100], "color": "#86EFAC"},
            ],
            "threshold": {
                "line": {"color": "#166534", "width": 3},
                "thickness": 0.8,
                "value": prob_lpd * 100,
            },
        },
    ))

    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=20, b=20),
        font=dict(family="Inter, system-ui", size=11),
    )
    return fig


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float) -> go.Figure:
    """Create ROC curve from FPR/TPR arrays."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {auc:.3f})',
        line=dict(color='#0284C7', width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='#64748B', width=1, dash='dash'),
    ))
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=350,
        margin=dict(l=60, r=40, t=60, b=40),
        plot_bgcolor="white",
        showlegend=True,
    )
    fig.update_xaxes(gridcolor="#E2E8F0", range=[0, 1])
    fig.update_yaxes(gridcolor="#E2E8F0", range=[0, 1])
    return fig


def plot_stroke_width_distribution(binary_mask: np.ndarray) -> go.Figure:
    """Create histogram of stroke width from distance transform."""
    import cv2
    dist_map = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated_dist = cv2.dilate(dist_map, kernel)
    ridge_mask = (dist_map == dilated_dist) & (dist_map > 0.0)

    if np.any(ridge_mask):
        local_radii = dist_map[ridge_mask].astype(np.float64)
        stroke_widths = 2.0 * local_radii

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=stroke_widths,
            nbinsx=30,
            marker_color="#0284C7",
            opacity=0.7,
            name="Stroke Width",
            hovertemplate="Width: %{x:.2f} px<br>Count: %{y}<extra></extra>",
        ))

        mean_sw = np.mean(stroke_widths)
        median_sw = np.median(stroke_widths)
        fig.add_vline(x=mean_sw, line_dash="dash", line_color="#D97706", annotation_text=f"Mean: {mean_sw:.2f}")
        fig.add_vline(x=median_sw, line_dash="dot", line_color="#16A34A", annotation_text=f"Median: {median_sw:.2f}")

        fig.update_layout(
            title="Stroke Width Distribution (Medial Axis Ridges)",
            xaxis_title="Stroke Width (pixels)",
            yaxis_title="Frequency",
            height=300,
            margin=dict(l=60, r=40, t=60, b=50),
            plot_bgcolor="white",
            font=dict(family="Inter, system-ui", size=11),
            showlegend=False,
        )
        fig.update_xaxes(gridcolor="#E2E8F0")
        fig.update_yaxes(gridcolor="#E2E8F0")
        return fig
    else:
        fig = go.Figure()
        fig.add_annotation(text="No stroke ridges detected", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=200)
        return fig


def plot_component_size_distribution(binary_mask: np.ndarray) -> go.Figure:
    """Create histogram of connected component areas."""
    import cv2
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
        areas = areas[areas >= 5]  # Filter noise

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=areas,
            nbinsx=30,
            marker_color="#8B5CF6",
            opacity=0.7,
            name="Component Area",
            hovertemplate="Area: %{x:.0f} px²<br>Count: %{y}<extra></extra>",
        ))

        fig.update_layout(
            title="Connected Component Area Distribution",
            xaxis_title="Area (pixels²)",
            yaxis_title="Frequency",
            height=300,
            margin=dict(l=60, r=40, t=60, b=50),
            plot_bgcolor="white",
            font=dict(family="Inter, system-ui", size=11),
            showlegend=False,
        )
        fig.update_xaxes(gridcolor="#E2E8F0")
        fig.update_yaxes(gridcolor="#E2E8F0")
        return fig
    else:
        fig = go.Figure()
        fig.add_annotation(text="No components found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=200)
        return fig


def plot_metrics_comparison(eval_metrics: pd.DataFrame) -> go.Figure:
    """Create grouped bar chart comparing model configurations."""
    if eval_metrics is None or len(eval_metrics) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Evaluation metrics not available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    metrics = ["accuracy_mean", "balanced_accuracy_mean", "precision_mean", "recall_pd_mean", "recall_lpd_mean", "f1_score_mean", "roc_auc_mean"]
    metric_labels = ["Accuracy", "Bal. Acc.", "Precision", "Recall (PD)", "Recall (LPD)", "F1-Score", "ROC-AUC"]

    fig = go.Figure()
    colors = ["#0284C7", "#F59E0B", "#16A34A", "#8B5CF6"]

    for idx, row in eval_metrics.iterrows():
        vals = [row[m] for m in metrics]
        errs = [row[m.replace("_mean", "_std")] for m in metrics]
        fig.add_trace(go.Bar(
            name=row["configuration"],
            x=metric_labels,
            y=vals,
            error_y=dict(type="data", array=errs, visible=True, color="rgba(0,0,0,0.3)"),
            marker_color=colors[idx % len(colors)],
            hovertemplate="<b>%{fullData.name}</b><br>%{x}: %{y:.3f} ± %{error_y.array:.3f}<extra></extra>",
        ))

    fig.update_layout(
        barmode="group",
        title="Model & Feature Set Comparison (5-Fold Stratified CV)",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.05], gridcolor="#E2E8F0"),
        xaxis=dict(gridcolor="#E2E8F0"),
        height=400,
        margin=dict(l=60, r=40, t=60, b=80),
        plot_bgcolor="white",
        font=dict(family="Inter, system-ui", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_gauge_chart(value: float, title: str, color: str, threshold_color: str) -> go.Figure:
    """Create a single gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={"text": title, "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#E2E8F0",
            "steps": [
                {"range": [0, 33], "color": "#FEF3C7"},
                {"range": [33, 66], "color": "#FDE68A"},
                {"range": [66, 100], "color": "#FCD34D"},
            ],
            "threshold": {
                "line": {"color": threshold_color, "width": 3},
                "thickness": 0.8,
                "value": value * 100,
            },
        },
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=20, b=20),
        font=dict(family="Inter, system-ui", size=11),
    )
    return fig