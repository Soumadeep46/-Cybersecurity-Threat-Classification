# Cybersecurity Threat Detection Using Machine Learning

## Overview
This project is focused on building a ** Cybersecurity Network Intrusion Detection System (NIDS)** using machine learning models to classify threats in network traffic. The system utilizes various machine learning models to detect potential threats, leveraging publicly available datasets for training and evaluation.

The **CICIDS2017 dataset** is used as the primary data source, which contains real-world attack scenarios and normal traffic.

### Dataset Used
**CICIDS2017 Dataset**: Available at [CIC IDS 2017](https://www.unb.ca/cic/datasets/ids.html)

## Features
- Multiple **machine learning models** for threat classification
- **Feature importance analysis** using SHAP and model-specific importance metrics
- **Performance evaluation** through confusion matrices and accuracy comparisons
- **Model stacking** to improve classification performance
- **Visualization of results** including feature importance and model comparisons

---
## Models Used
This project implements and evaluates the following machine learning models:

1. **Logistic Regression** - A simple baseline model used for comparison.
2. **Random Forest Classifier** - An ensemble learning method based on decision trees.
3. **XGBoost Classifier** - A gradient-boosting framework optimized for performance.
4. **Stacking Classifier** - Combines multiple models to improve accuracy.

Each model is trained and tested using the CICIDS2017 dataset to detect network threats.

---
## Installation
Ensure you have Python installed, then install all required dependencies with:

```bash
pip install -r requirements.txt
```

Alternatively, install dependencies directly with:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn shap joblib
```

---
## Usage
To train and evaluate the models, run:

```bash
python threat_classifier.py
```

This script will:
1. Load and preprocess the dataset
2. Train each machine learning model
3. Evaluate performance using accuracy, precision, recall, and confusion matrices
4. Generate visualizations for analysis

---
## Output Files
Running the script will generate the following output files:

### Model Performance Metrics
- **Confusion Matrices**
  - `confusion_matrix_logistic_regression.png`
  - `confusion_matrix_random_forest.png`
  - `confusion_matrix_xgboost.png`
  - `confusion_matrix_stacking.png`

- **Feature Importance Analysis**
  - `feature_importance_random_forest.png`
  - `feature_importance_xgboost.png`
  - `feature_importance_alt_random_forest.png`
  - `shap_summary_random_forest.png`

- **Model Comparison**
  - `model_comparison.png`

- **Serialized Model Files**
  - `random_forest_20250326_000904.joblib`
  - `feature_names_20250326_000904.joblib`

- **Report File**
  - `report_20250326_000904.txt`

---
## Results and Observations
- **Random Forest and XGBoost** provided high accuracy in identifying network threats.
- **Stacking Classifier** improved overall performance by combining multiple models.
- Feature importance analysis helped in identifying key network traffic features influencing predictions.
- The confusion matrices highlighted the classification performance for each model.

---
## Future Enhancements
- Integrate **deep learning models** (e.g., LSTMs) for sequence-based network traffic analysis.
- Optimize hyperparameters using **automated tuning** techniques.
- Deploy the trained model as a **real-time intrusion detection system**.



Feel free to contribute, suggest improvements, or raise issues!

---
## License
This project is released under the **MIT License**.

