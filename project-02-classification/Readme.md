### 2. For Project 2 (KNN Classification Engine)
```markdown
# 🌸 DecodeLabs AI Project 2: KNN Classification Engine

**Author:** Sandeep Kumar Sidar  
**Institution:** International Institute of Information Technology (IIIT), Naya Raipur  
**Batch:** 2026 | DecodeLabs Artificial Intelligence Track

---

## 📌 Overview
This project transitions from hardcoded rules to supervised machine learning. It features a custom K-Nearest Neighbors (KNN) classification engine built to accurately categorize biological classes within the classic Iris dataset based on physical feature measurements.

## 🚀 Key Features & Pipeline
* **Data Splitting:** Robust 80/20 train-test split (`test_size=0.2`) with randomized shuffling (`shuffle=True`) to eliminate ordering bias.
* **Feature Scaling (Crucial):** Implementation of `StandardScaler` to normalize variance across features. The scaler is strictly fitted only on the training data to prevent data leakage.
* **Model Optimization:** Trained using the `KNeighborsClassifier` with an optimal neighbor count ($K=5$, the Elbow Point) to balance underfitting and overfitting.
* **Detailed Validation:** Evaluates model performance using Accuracy, Weighted F1-Scores, Confusion Matrices, and a full Classification Report.
* **Live Custom Inference:** Includes a live terminal input module allowing users to input custom flower measurements and instantly receive a predicted class along with probability confidence scores (`predict_proba`).

## 🛠️ Tech Stack
* **Language:** Python
* **Machine Learning:** Scikit-Learn
* **Algorithms:** K-Nearest Neighbors (KNN), StandardScaler

## 🏃‍♂️ How to Run
Execute the script to train the model, view metrics, and test custom measurements:
```bash
python knn_classifier.py