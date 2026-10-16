# 📈 Financial Trading Classification & Stacked Ensemble Pipeline

An advanced, end-to-end machine learning pipeline designed for financial time-series classification. This project implements a robust stacking architecture combining gradient boosting models (CatBoost, XGBoost, LightGBM) and Logistic Regression, augmented with unsupervised market regime detection (Gaussian Mixture Models), SHAP-based feature selection, and polynomial feature interactions.

---

## 🏗️ Architecture & Pipeline Workflow

1. **Temporal Splitting:** Data is strictly split by time (training/validation prior to `2024-01-01`; out-of-sample testing from `2024-01-01` onward) to prevent look-ahead bias and data leakage.
2. **Feature Engineering & Regime Detection:**
   * **Initial SHAP Screening:** Selects the top 50 features using an initial CatBoost model.
   * **Market Regimes (GMM):** Scales the top features and fits a **Gaussian Mixture Model (GMM)** to cluster market states into distinct regimes (added as a categorical `Regime` feature).
   * **Final SHAP Reduction:** Re-evaluates feature importance across the dataset + `Regime` to select the most predictive feature subset.
   * **Feature Interactions:** Generates multiplicative pairwise interaction terms to capture non-linear market relationships.
3. **Hyperparameter Tuning:** Uses `RandomizedSearchCV` combined with `TimeSeriesSplit` cross-validation (3 splits) optimized for F1-score.
4. **Stacked Generalization (Meta-Learner):**
   * Base models (CatBoost, XGBoost, LightGBM, and Logistic Regression) generate probability predictions on the validation set.
   * A final **Logistic Regression meta-learner** combines these probabilities to make the ultimate classification decision.
5. **Custom Threshold Optimization & Stability Scoring:**
   * Sweeps across classification thresholds ($0.01$ to $0.99$) to maximize the F1-score while filtering out statistically insignificant sample sizes (< 10 instances for FP/FN/TN).
   * Computes a custom **Stability Score** to measure trade reliability against baseline class proportions.

---

## 🛠️ Tech Stack & Dependencies

* **Python 3.8+**
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`, `catboost`, `xgboost`, `lightgbm`
* **Explainability:** `shap`

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/waseemshammout/trading_model.git
cd trading_model
```

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn catboost xgboost lightgbm shap
```

### 3. Prepare Your Data
Ensure your dataset is placed in the root directory as `SE5.csv`. It must contain:
* A `Time` column (datetime format).
* A binary target column named `class` (where positive values represent profitable setups/desired market directions).
* Raw feature columns.

### 4. Run the Pipeline
Execute the main script to train the base models, tune hyperparameters, fit the meta-learner, and evaluate performance on the test split:
```bash
python main.py
```

---

## 📊 Key Code Modules

* **`calculate_stability_score()`**: Computes a domain-specific metric evaluating true positive/false positive ratios relative to overall class frequencies.
* **`evaluate_model()`**: Performs threshold optimization with safety filters against misleading low-sample outputs.
* **`preprocess_data()` & `preprocess_test_data()`**: Encapsulate the entire feature transformation pipeline—ensuring parameters fitted on training data (GMM scalers, feature lists) are safely applied to validation and test sets without data leakage.
* **`train_base_model()`**: Manages cross-validated hyperparameter tuning and model training across CatBoost, XGBoost, LightGBM, and Logistic Regression.

---

## 📝 License
This project is open-source and available under the [MIT License](LICENSE).