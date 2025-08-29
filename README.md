# ML-Diabetes-Classifier - Machine Learning Based Diabetes Prediction System

## Overview

ML-Diabetes-Classifier is a machine learning project designed to predict the likelihood of diabetes in patients using a range of advanced classification algorithms. The system leverages several popular machine learning models and provides a comprehensive comparison using both tabular metrics and visualizations.

---

## Features

- Trains and evaluates multiple ML models on a diabetes dataset
- Compares model performance using accuracy, precision, recall, F1-score, and ROC-AUC
- Predicts diabetes risk for new patient data across all models
- Visualizes results with ROC curves, confusion matrices, and comparison bar charts
- Generates model comparison tables for easy interpretation

---

## Machine Learning Models Used

- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Multi-Layer Perceptron (MLP / Neural Network)**
- **Decision Tree**
- **Random Forest**
- **Gradient Boosting (e.g., XGBoost or GradientBoostingClassifier)**

Each model is trained, evaluated, and compared on the same dataset, enabling robust analysis.

---

## Example Results

### Model Performance Metrics

| Model               | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.79     | 0.81      | 0.77   | 0.79     | 0.84    |
| KNN                 | 0.77     | 0.78      | 0.75   | 0.76     | 0.81    |
| SVM                 | 0.80     | 0.82      | 0.78   | 0.80     | 0.85    |
| MLP                 | 0.81     | 0.83      | 0.79   | 0.81     | 0.86    |
| Decision Tree       | 0.75     | 0.76      | 0.73   | 0.74     | 0.78    |
| Random Forest       | 0.82     | 0.84      | 0.80   | 0.82     | 0.87    |
| Gradient Boosting   | 0.83     | 0.85      | 0.81   | 0.83     | 0.88    |

> *Note: The above metrics are example results. Actual results may vary depending on dataset splits and hyperparameters.*

---

### Visualizations & Charts

The notebook provides the following charts for analysis:

#### 1. **ROC Curves for All Models**
- Shows the trade-off between sensitivity and specificity for each classifier.
![Image Alt](https://github.com/Debraj-80/ML-Diabetes-Classifier/blob/8cbf07b0f141c4b283948343c63ff6e28f7746a3/assets/3.png)

#### 2. **Bar Charts for Metric Comparison**
- Compares accuracy, precision, recall, and F1-score across all models.
![Image Alt](https://github.com/Debraj-80/ML-Diabetes-Classifier/blob/8cbf07b0f141c4b283948343c63ff6e28f7746a3/assets/4.png)
> **Tip:** The notebook auto-generates these charts after training and evaluating the models.

---

### Example Workflow

- Load and inspect the dataset
- Split data into training and testing sets
- Scale features for better model performance
- Train multiple ML models and evaluate each using accuracy, ROC-AUC, confusion matrix, and classification report
- Predict diabetes risk for new patients and visualize the results

---

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook or Google Colab
- Required Python libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - xgboost (for Gradient Boosting)
  - prettytable

### Setup & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Debraj-80/ML-Diabetes-Classifier.git
   cd ML-Diabetes-Classifier
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost prettytable
   ```

3. **Run the notebook:**
   - Open `diabetes_classifier.ipynb` in Jupyter Notebook or Google Colab.
   - Follow the instructions in the notebook cells to run the code.

4. **Dataset:**
   - The notebook expects a file named `diabetes.csv` with diabetes dataset features and target variable.

---

## Important Details

- Each algorithm's predictions are compared and displayed for both model evaluation and new patient prediction.
- Model performance metrics and probability outputs are shown in tables and plots for transparency.
- The system is intended for educational and research purposes and is not suitable for real-world clinical use without further validation.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

- [Kaggle - Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- [scikit-learn](https://scikit-learn.org/)
