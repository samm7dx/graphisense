Key Commands
Environment Setup
# Create virtual environment
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# For CUDA (optional): https://pytorch.org/get-started/locally/
Dataset
# Download from: https://data.mendeley.com/datasets/39hr8dx76p/1
# Extract to data/raw/ so structure is:
data\raw\
├── Potential Dysgraphia\
└── Low Potential Dysgraphia\
Random Forest (Basic Analysis)
# Feature extraction, analysis, and RF training + 5-fold CV
python -m src.features.build_feature_dataset
python -m src.features.analyze_features
python -m src.models.train_evaluate
→ Outputs: outputs/models/model_pipeline.joblib, outputs/models/model_metadata.json
CNN/ResNet-18 (Advanced Analysis)
# Train (50 epochs, early stopping)
python -m src.models.cnn_train --dataset-root data/raw --output-dir outputs/models/cnn --epochs 50 --batch-size 16 --num-workers 0

# Evaluate on held-out test split
python -m src.models.cnn_evaluate --checkpoint outputs/models/cnn/checkpoints/best_model.pt --dataset-root data/raw --output-dir outputs/reports/cnn_evaluation
→ Outputs: outputs/models/cnn/checkpoints/best_model.pt, outputs/reports/cnn_evaluation/evaluation_results.json
Run Tests
python -m pytest tests/ -q
Launch Streamlit App
python -m streamlit run app/main.py
→ Opens http://localhost:8501
Programmatic Inference
# Random Forest
from src.models.inference import predict_handwriting
result = predict_handwriting(Path("data/raw/Potential Dysgraphia/PD (1).jpg"))

# CNN/ResNet-18
from src.models.cnn_inference import predict_handwriting_cnn
result = predict_handwriting_cnn(Path("data/raw/Potential Dysgraphia/PD (1).jpg"))
