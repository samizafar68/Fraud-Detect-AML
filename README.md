# 🔐 Fraud Detection and Anyslsis using Machine Learning

The main goal is to build an end-to-end machine learning pipeline that can detect fraudulent financial transactions through exploratory data analysis, preprocessing, classification, and evaluation.

---


## 🎯 Objective

To analyze transaction data and develop multiple classification models capable of identifying fraudulent activities (`isFraud`) using structured ML practices, including:
- Data cleaning and transformation
- Feature selection and normalization
- Model tuning and evaluation
- Outlier detection and handling

---

## 📊 Part 1: Exploratory Data Analysis

- Loaded and inspected dataset
- Handled missing values using median (numerical) and mode (categorical)
- Visualized distributions with:
  - Histograms
  - Count plots
  - Boxplots
  - Violin plots
- Outlier detection using:
  - **IQR Method**
  - **Z-score**
  - **Isolation Forest (unsupervised anomaly detection)**
- Capped and log-transformed skewed features
- Analyzed:
  - Feature relationships with `isFraud`
  - Correlation matrix
  - Skewness and kurtosis
- Summarized statistical insights

---

## 🧹 Part 2: Data Preprocessing & Splitting

- Removed or transformed outliers
- One-hot encoded `transactionType`
- Applied both:
  - **Min-Max Normalization**
  - **Z-score Standardization**
- Selected top 8 features using:
  - `SelectKBest` with **ANOVA F-test**
- Performed stratified train-test split (80-20)

---

## 🤖 Part 3: Model Implementation

### ✅ K-Nearest Neighbors (KNN)
- Used different distance metrics: Euclidean, Manhattan, Minkowski
- Tuned `k` values and evaluated accuracy
- Saved best model using `joblib`

### ✅ Support Vector Machines (SVM)
- Linear, RBF kernel tested
- Tuning of `C` parameter (regularization)
- Used both `SVC` and `SGDClassifier` variants

### ✅ Decision Trees
- Tuned `max_depth`, `min_samples_split`, and `min_samples_leaf`
- Visualized the decision tree using `plot_tree`
- Saved model for future prediction

### ✅ Logistic Regression
- Applied L1 (Lasso) and L2 (Ridge) regularization
- Evaluated all variations
- Saved final model

---

## 📊 Part 4: Evaluation & Comparison

- Evaluation metrics:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-Score**
- Classification reports for all models
- Confusion matrices with heatmaps
- Computation time for:
  - Training
  - Prediction
- Class imbalance awareness (`isFraud` is minority class)

---


## 🛠️ Tech Stack

- **Language**: Python
- **Tools**: Jupyter Notebook
- **Libraries**:
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `scikit-learn`, `joblib`

---

## 📁 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/AML_Fraud_Detection.git
   cd AML_Fraud_Detection
