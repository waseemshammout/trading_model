import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import shap
import catboost as cat 
import xgboost as xgb
import lightgbm as lgb
from datetime import timedelta, datetime
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('SE5.csv')
df['Time'] = pd.to_datetime(df['Time'])
df = df.sort_values(by='Time', ascending=True)

train_val_df = df[df["Time"] < "2024-01-01"]
test_df = df[df["Time"] >= "2024-01-01"]

X = train_val_df.drop(columns=['class'])
y = train_val_df['class']

print(f"Train/Validation data shape: {train_val_df.shape}")
print(f"Test data shape: {test_df.shape}")

def calculate_stability_score(y_true, pred, class1_proportion):
    """
    Function to calculate stability using true values, probabilities and profitable trades proportion
    """
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    if fp == 0:
        return np.inf if tp > 0 else 0.0
    if class1_proportion == 0:
        return 0.0
    stability = (tp / fp) / class1_proportion
    return stability

###################################################################################################################

def evaluate_model(y_true, y_prob, model_name, class1_proportion_val, thresholds=None):
    """
    Simplified evaluation function with built-in optimal threshold finding,
    with a filter to avoid selecting unrealistic 100% precision/recall.
    Input: True values, probability predictions, profitable trades proportion, explicit thresholds (optional)
    Returns metrics dictionary.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 100) # a range of 100 threshold option
        
    thresholds = [t for t in thresholds if 0.001 < t < 0.999] # sanity check
    results = []
    
    # Find optimal threshold
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        # Get confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate evaluation metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        stability = calculate_stability_score(y_true, y_pred, class1_proportion_val)
        
        # Saving metrics and confusion matrics components for later filtering
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'stability': stability,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp 
        })

    if not results:  # Fallback if no thresholds
        # This part remains as a safety fallback
        threshold = 0.5
        y_pred = (y_prob >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        stability = calculate_stability_score(y_true, y_pred, class1_proportion_val)
        cm = confusion_matrix(y_true, y_pred)
    else:
        results_df = pd.DataFrame(results)

        # This to consider options where false positive, false negative and true negative > 10 instances otherwise it's misleading.
        misleading_options = (results_df['fp'] < 10) | (results_df['fn'] < 10) | (results_df['tn'] < 10)
        
        filtered_df = results_df[~(misleading_options)].copy()

        # Fallback if filtering removes all candidates (if all predictions are extreme), use original results if all are deemed extreme
        if filtered_df.empty:
            print(f"Warning: All candidates filtered out for {model_name}. Reverting to original selection without extreme filtering.")
            filtered_df = results_df.copy()
        else:
            print(f"Extreme filtering is done, original results {results_df.shape[0]} row, filtered {filtered_df.shape[0]}")

        # Find best threshold from the filtered (or original) candidates
        # If more than instance have the same max F1, we take the higher Recall
        max_f1 = filtered_df['f1'].max()
        best_candidates = filtered_df[filtered_df['f1'] == max_f1]
        best_row = best_candidates.loc[best_candidates['recall'].idxmax()]
        
        threshold = best_row['threshold']
        precision = best_row['precision']
        recall = best_row['recall']
        f1 = best_row['f1']
        stability = best_row['stability']
        
        # Re-calculate confusion matrix for the final selected optimal threshold
        y_pred = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
    return {
        'model': model_name,
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'stability': stability,
        'confusion_matrix': cm
    }

###################################################################################################################

n_features = 715
def preprocess_data(train, valid=None, top_features_ratio=0.03, n_regimes=3):

    """
    This function does the following:
    1. SHAP-Based Feature Selection to select top 50 most important features for market regime identification.
    2. Apply un-supervised learning using GMM to predict regime.
    3. SHAP-Based Feature Selection on the whole features + 'Regime' feature, selecting top x% important ones based on top_features_ratio argument
    4. Apply multiplicative interaction on the selected top x% features (excluding regime)
    Returns 
        a. Engineered features and labels for training and validation set (where applicable)
        b. The fitted objects used for scaling features and regime predictions
        c. Top features as list
        Note: b & c will be used for applying on test data.
    """

    # Assigning features and labels variables
    X_train, y_train = train.drop(columns=['Time','class']), train['class']
    X_valid, y_valid = valid.drop(columns=['Time','class']), valid['class']
    
    # Class weight = negative class n / negative class n
    class_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    
    # Step 1 indicated above
    tmp_model_initial_shap = cat.CatBoostClassifier(random_state=1, scale_pos_weight=class_weight, loss_function='Logloss', eval_metric='F1', early_stopping_rounds=5, verbose=0)
    tmp_model_initial_shap.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    explainer_initial = shap.TreeExplainer(tmp_model_initial_shap)
    shap_values_initial = explainer_initial.shap_values(X_train)
    features_shap_initial = pd.DataFrame({'Feature' : X_train.columns,'Importance' : np.abs(shap_values_initial).mean(axis=0)}).sort_values(by='Importance', ascending=False)
    gmm_input_features = list(features_shap_initial['Feature'].head(50).values)

    # Step 2
    gmm_data_tr = X_train[gmm_input_features]
    gmm_data_va = X_valid[gmm_input_features]
    # Scale features for GMM
    scaler_gmm = StandardScaler()
    gmm_data_tr_scaled = scaler_gmm.fit_transform(gmm_data_tr)
    gmm_data_va_scaled = scaler_gmm.transform(gmm_data_va)
    # Train GMM and predict regimes
    try:
        gmm = GaussianMixture(n_components=n_regimes, random_state=1, covariance_type='full', max_iter=200, tol=1e-3)
        gmm.fit(gmm_data_tr_scaled)
        regime_labels_tr = gmm.predict(gmm_data_tr_scaled)
        regime_labels_va = gmm.predict(gmm_data_va_scaled)
    except ValueError as e: # sometimes GMM fails so we assign regime 0
        print(f"Warning: GMM failed with error: {e}. Assigning default regime 0.")
        regime_labels_tr = np.zeros(len(X_train), dtype=int)
        regime_labels_va = np.zeros(len(X_valid), dtype=int)
    # Add regime labels
    X_train['Regime'] = regime_labels_tr
    X_valid['Regime'] = regime_labels_va
    # Convert regime column to 'category'
    X_train['Regime'] = X_train['Regime'].astype('category')
    X_valid['Regime'] = X_valid['Regime'].astype('category')
    
    # Step 3
    tmp_model_final_shap = cat.CatBoostClassifier(random_state=1, scale_pos_weight=class_weight, loss_function='Logloss', eval_metric='F1', early_stopping_rounds=10, verbose=0)
    tmp_model_final_shap.fit(X_train, y_train, cat_features=[len(X_train.columns)-1], eval_set=[(X_valid, y_valid)])
    explainer_final = shap.TreeExplainer(tmp_model_final_shap)
    shap_values_final = explainer_final.shap_values(X_train)
    features_shap_final = pd.DataFrame({'Feature':X_train.columns,'Importance':np.abs(shap_values_final).mean(axis=0)}).sort_values(by='Importance', ascending=False)
    top_features = list(features_shap_final.head(int(n_features * top_features_ratio))['Feature'].values)
    X_train_selected = X_train[top_features]
    X_valid_selected = X_valid[top_features]

    # Step 4
    X_train_interacted = X_train_selected.copy()
    X_valid_interacted = X_valid_selected.copy()
    features_for_interaction = [f for f in top_features if f != 'Regime'] # exclude regime column from interaction since it's categorical

    for i in range(len(features_for_interaction)):
        for j in range(i + 1, len(features_for_interaction)):
            feature1 = features_for_interaction[i]
            feature2 = features_for_interaction[j]
            X_train_interacted[f'{feature1}_x_{feature2}'] = X_train_interacted[feature1] * X_train_interacted[feature2]
            X_valid_interacted[f'{feature1}_x_{feature2}'] = X_valid_interacted[feature1] * X_valid_interacted[feature2]

    return X_train_interacted, y_train, X_valid_interacted, y_valid, scaler_gmm, gmm, top_features

###################################################################################################################

# For hyperparameters tuning we have 3 param_distribution for search per model (Tree-based only)
params_search_cat = {'depth': [4, 6, 8, 10],'learning_rate': [0.1, 0.05, 0.01, 0.005],'l2_leaf_reg': [1, 3, 5, 7, 9, 11],'bagging_temperature': [0, 0.5, 1, 3, 5],'subsample': [0.5, 0.7, 0.9, 1],'random_strength': [1e-9, 1e-6, 1e-3, 1e-1],'border_count': [32, 64, 128, 254]}
params_search_xgb = {'learning_rate': [0.1, 0.05, 0.01, 0.005],'max_depth': [3, 6, 10],'min_child_weight': [1, 3, 5],'subsample': [0.6, 0.8, 1.0],'colsample_bytree': [0.6, 0.8, 1.0],'gamma': [0, 0.1, 0.5],'reg_alpha': [0, 0.1, 1],'reg_lambda': [1, 5, 10]}
params_search_lgb = {'learning_rate': [0.1, 0.05, 0.01, 0.005],'max_depth': [3, 5, 7, 10, -1],'num_leaves': [15, 31, 63, 127],'min_child_samples': [10, 20, 30, 50],'subsample': [0.6, 0.8, 1.0],'colsample_bytree': [0.6, 0.8, 1.0],'reg_alpha': [0, 0.01, 0.1, 1],'reg_lambda': [0, 0.01, 0.1, 1],}

###################################################################################################################

def train_base_model(X_train, y_train, X_valid=None, y_valid=None, model_name='CatBoost'):

    """
    This function:
    1. Use param_search_xxx to tune hyperparameter the base model before fitting
    2. Works according to input model_name argument ['CatBoost','XGBoost','LogReg','LGBM']
    3. Predict probabilities for the selected model
    Retruns:
        a. Probability predictions
        b. model as object
        c. Feature scaler as object (only for LogReg)
        Note: b & c will be used for applying on test data.
    """
    
    class_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    feature_scaler = None
    print(f'Hyperparameter tuning, base model: {model_name}, {datetime.now()}')
    if model_name == 'CatBoost':
        model = cat.CatBoostClassifier(random_state=1, scale_pos_weight=class_weight, loss_function='Logloss',  
                                       eval_metric='F1', early_stopping_rounds=50, iterations=1000, verbose=0)
        def cat_features_regime(cols):
            # This function is dedicated for catboost model.fit() to accomodate for Regime column as categorical feature
            if 'Regime' in cols:
                return ['Regime']
            return None
        search = RandomizedSearchCV(estimator=model, param_distributions=params_search_cat, n_iter=10, cv=TimeSeriesSplit(n_splits=3), random_state=1, scoring='f1', verbose=0, n_jobs=-1)
        search.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], cat_features=cat_features_regime(list(X_train.columns)))
    elif model_name == 'XGBoost':
        model = xgb.XGBClassifier(objective='binary:logistic', scale_pos_weight=class_weight, random_state=1, n_estimators=1000, early_stopping_rounds=10, enable_categorical=True, tree_method='hist', verbosity=0)
        search = RandomizedSearchCV(estimator=model, param_distributions=params_search_xgb, n_iter=10, cv=TimeSeriesSplit(n_splits=3), random_state=1, scoring='f1', verbose=0, n_jobs=-1)
        search.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    elif model_name == 'LogReg':
        feature_scaler = StandardScaler()
        x_tr_scaled = feature_scaler.fit_transform(X_train)
        model = LogisticRegression(max_iter=1000, random_state=1, solver='liblinear', class_weight={0: 1, 1: class_weight}, C=0.01, penalty='l1')
        model.fit(x_tr_scaled, y_train)
    elif model_name == 'LGBM':
        model = lgb.LGBMClassifier(class_weight='balanced', objective='binary', random_state=1, learning_rate= 0.01, verbosity=-1, n_estimators=1000)
        search = RandomizedSearchCV(estimator=model, param_distributions=params_search_lgb, n_iter=10, cv=TimeSeriesSplit(n_splits=3), random_state=1, scoring='f1', verbose=0, n_jobs=-1)
        search.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric="binary_logloss", callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)])

    if model_name != 'LogReg':
        model = search.best_estimator_

    print(f'Base model {model_name} is completed, {datetime.now()}')
    try:
        prob = model.predict_proba(X_valid)[:, 1]
    except Exception as e:
        print(f"{model} failed: {e}. Filling with zeros.")
        prob = np.zeros(len(y_valid))

    return prob, model, feature_scaler

###################################################################################################################

regimes = 3

# Rolling Window Cross-Validation
train_size_days = 550
valid_size_days = 90

train_period_list = list()
valid_period_list = list()

precision_list = list()
recall_list = list()
f1_list = list()
stability_list = list()
conf_matrix_list = list()
threshold_list = list()

fold = 1 # initiated
start_date_cv = train_val_df['Time'].min()

while True:
    train_start = start_date_cv
    train_end = train_start + timedelta(days=train_size_days)
    val_start = train_end + timedelta(minutes=5)
    val_end = val_start + timedelta(days=valid_size_days)
    if val_end > train_val_df["Time"].max():
        break
    train_set = train_val_df[(train_val_df["Time"] >= train_start) & (train_val_df["Time"] < train_end)]
    valid_set = train_val_df[(train_val_df["Time"] >= val_start) & (train_val_df["Time"] < val_end)]

    print(f"Fold {fold}")
    print(f"Training Period: {train_start.strftime('%Y-%m-%d')} - {train_end.strftime('%Y-%m-%d')} | Validation Period: {val_start.strftime('%Y-%m-%d')} - {val_end.strftime('%Y-%m-%d')}")
    train_period_list.append(f'{train_start.strftime('%Y-%m')} - {train_end.strftime('%Y-%m')}')
    valid_period_list.append(f'{val_start.strftime('%Y-%m')} - {val_end.strftime('%Y-%m')}')

    # Preprocess data
    X_train_interacted, y_train, X_valid_interacted, y_valid, scaler_gmm, gmm, top_features = preprocess_data(train_set, valid_set, top_features_ratio=0.02, n_regimes=regimes)

    # Training base models and getting probabilities
    cat_prob, _, _ = train_base_model(X_train_interacted, y_train, X_valid_interacted, y_valid, 'CatBoost')
    xgb_prob, _, _ = train_base_model(X_train_interacted, y_train, X_valid_interacted, y_valid, 'XGBoost')
    lr_prob, _, _ = train_base_model(X_train_interacted, y_train, X_valid_interacted, y_valid, 'LogReg')
    lgb_prob, _, _ = train_base_model(X_train_interacted, y_train, X_valid_interacted, y_valid, 'LGBM')

    ############################################ Evaluation on validation data ###############################################

    class1_ratio_val = y_valid.value_counts(normalize=True)[1]

    cat_metrics = evaluate_model(y_valid, cat_prob, "CatBoost", class1_ratio_val)
    xgb_metrics = evaluate_model(y_valid, xgb_prob, "XGBoost", class1_ratio_val)
    lr_metrics = evaluate_model(y_valid, lr_prob, "LogReg", class1_ratio_val)
    lgb_metrics = evaluate_model(y_valid, lgb_prob, "LGBM", class1_ratio_val)

    precision_list.append(
        {
            "CatBoost": cat_metrics["precision"],
            "XGBoost": xgb_metrics["precision"],
            "LogReg": lr_metrics["precision"],
            "LGBM": lgb_metrics["precision"],
        }
    )

    recall_list.append(
        {
            "CatBoost": cat_metrics["recall"],
            "XGBoost": xgb_metrics["recall"],
            "LogReg": lr_metrics["recall"],
            "LGBM": lgb_metrics["recall"],
        }
    )

    f1_list.append(
        {
            "CatBoost": cat_metrics["f1"],
            "XGBoost": xgb_metrics["f1"],
            "LogReg": lr_metrics["f1"],
            "LGBM": lgb_metrics["f1"],
        }
    )

    stability_list.append(
        {
            "CatBoost": cat_metrics["stability"],
            "XGBoost": xgb_metrics["stability"],
            "LogReg": lr_metrics["stability"],
            "LGBM": lgb_metrics["stability"],
        }
    )

    conf_matrix_list.append(
        {
            "CatBoost": cat_metrics["confusion_matrix"],
            "XGBoost": xgb_metrics["confusion_matrix"],
            "LogReg": lr_metrics["confusion_matrix"],
            "LGBM": lgb_metrics["confusion_matrix"],
        }
    )

    threshold_list.append(
        {
            "CatBoost": cat_metrics["threshold"],
            "XGBoost": xgb_metrics["threshold"],
            "LogReg": lr_metrics["threshold"],
            "LGBM": lgb_metrics["threshold"],
        }
    )
    f_metrics = pd.DataFrame([cat_metrics, xgb_metrics, lr_metrics, lgb_metrics])
    print(f_metrics.round(3).to_string(max_colwidth=50), '\n')
    start_date_cv = val_end
    fold += 1

recalls = pd.DataFrame(recall_list)
precisions = pd.DataFrame(precision_list)
f1s = pd.DataFrame(f1_list)
stabilities = pd.DataFrame(stability_list)
thresholds = pd.DataFrame(threshold_list)
cms = pd.DataFrame(conf_matrix_list)

results = pd.DataFrame(
    {
        "Regime": [regimes] * 4,
        "Mean_Threshold": thresholds.mean(),
        "Mean_Recall": recalls.mean(),
        "Mean_Precision": precisions.mean(),
        "Mean_F1": f1s.mean(),
        "Mean_Stability": stabilities.mean(),
    }
)
print(results)