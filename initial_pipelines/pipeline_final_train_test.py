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
# Training base models on full data (or partial)
###################################################################################################################

# Input training start date and end date
# Validation start date: will be automatically set to the next day after training end date
# Validation end date: will be automatically set to the last row in 2023 data
# Input Regime count

input_train_start = '2022-01-01'
input_train_end   = '2022-12-31'
regimes = 3


tr_start_date = datetime.strptime(input_train_start, "%Y-%m-%d")
tr_end_date = datetime.strptime(input_train_end, "%Y-%m-%d")
va_start_date = tr_end_date + timedelta(days=1)
va_end_date = train_val_df["Time"].max()
training_data = train_val_df[(train_val_df["Time"] >= tr_start_date) & (train_val_df["Time"] <= tr_end_date)]
validation_data = train_val_df[(train_val_df["Time"] >= va_start_date) & (train_val_df["Time"] <= va_end_date)]
print(f"Training Period: {tr_start_date.strftime('%Y-%m-%d')} - {tr_end_date.strftime('%Y-%m-%d')} | Validation Period: {va_start_date.strftime('%Y-%m-%d')} - {va_end_date.strftime('%Y-%m-%d')}.")
# Apply data preprocessing
X_train_interacted, y_train, X_valid_interacted, y_valid, scaler_gmm, gmm, top_features = preprocess_data(training_data, validation_data, top_features_ratio=0.02, n_regimes=regimes)
# collect predictions from base model
cat_prob, cat_model, _ = train_base_model(X_train_interacted, y_train, X_valid_interacted, y_valid, model_name='CatBoost')
xgb_prob, xgb_model, _ = train_base_model(X_train_interacted, y_train, X_valid_interacted, y_valid, model_name='XGBoost')
lr_prob, lr_model, feature_scaler = train_base_model(X_train_interacted, y_train, X_valid_interacted, y_valid, model_name='LogReg')
lgb_prob, lgb_model, _ = train_base_model(X_train_interacted, y_train, X_valid_interacted, y_valid, model_name='LGBM')

###################################### Check validation metrics #########################################

class1_proportion = y_valid.value_counts(normalize=True)[1]

cat_metrics = evaluate_model(y_valid, cat_prob, "CatBoost", class1_proportion)
xgb_metrics = evaluate_model(y_valid, xgb_prob, "XGBoost", class1_proportion)
lr_metrics = evaluate_model(y_valid, lr_prob, "LogReg", class1_proportion)
lgb_metrics = evaluate_model(y_valid, lgb_prob, "LGBM", class1_proportion)

full_data_metrics = pd.DataFrame([cat_metrics, xgb_metrics, lr_metrics, lgb_metrics])
print(full_data_metrics.round(3).to_string(max_colwidth=50), '\n')

###################################################################################################################
# Training the meta learner
###################################################################################################################

meta_X_train = pd.DataFrame(
    {
        "cat_prob": cat_prob,
        "xgb_prob": xgb_prob,
        "logreg_prob": lr_prob,
        "lgbm_prob": lgb_prob,
    }
)
meta_y_train = y_valid

final_meta_learner = LogisticRegression(max_iter=1000)
final_meta_learner.fit(meta_X_train, meta_y_train)

###################################################################################################################

def preprocess_test_data(test_df, start_date, end_date, scaler_gmm, gmm, top_features):

    """
    This function 
    1. We input the testing dataset, start data and end date of the interval in question.
    2. It applies the same preprocessing we made on the training data, using the output objects from preprocess_data function
    Returns engineered features and labels for the interval period from testing data
    """

    test_df = test_df[(test_df["Time"] >= start_date) & (test_df["Time"] <= end_date)]

    x_test = test_df.drop(columns=["Time", "class"])
    y_test = test_df["class"]

    scaled_x_test_array = scaler_gmm.transform(x_test[scaler_gmm.feature_names_in_]) # Apply the scaling on features 
    scaled_x_test_df = pd.DataFrame(data=scaled_x_test_array, columns=x_test[scaler_gmm.feature_names_in_].columns) # Convert scaled features to df
    test_regime_labels = gmm.predict(scaled_x_test_df) # Predict regime labels
    x_test['Regime'] = test_regime_labels # add regime labels to the raw df
    x_test_selected = x_test[top_features] # Align top features with training set
    # Applying features interaction on the selected ones
    x_test_interacted = x_test_selected.copy()
    features_for_interaction = [f for f in top_features if f != 'Regime']
    for i in range(len(features_for_interaction)):
            for j in range(i + 1, len(features_for_interaction)):
                feature1 = features_for_interaction[i]
                feature2 = features_for_interaction[j]
                x_test_interacted[f'{feature1}_x_{feature2}'] = x_test_interacted[feature1] * x_test_interacted[feature2]
    # Check the features alignment between training & testing sets
    assert list(X_train_interacted.columns) == list(x_test_interacted.columns)

    return x_test_interacted, y_test

###################################################################################################################
# See results on un-seen data
###################################################################################################################

# input interval start and end date
test_start_date = "2024-01-01"
test_end_end = "2024-03-31"

# Preprocess testing data
x_test_interacted, y_test = preprocess_test_data(test_df=test_df, start_date=test_start_date, end_date=test_end_end, scaler_gmm=scaler_gmm, gmm=gmm, top_features=top_features)

# Collect predictions from the fitted base models
cat_pred_test = cat_model.predict_proba(x_test_interacted)[:, 1]
xgb_pred_test = xgb_model.predict_proba(x_test_interacted)[:, 1]
lr_pred_test = lr_model.predict_proba(feature_scaler.transform(x_test_interacted))[:, 1]
lgbm_pred_test = lgb_model.predict_proba(x_test_interacted)[:, 1]

# Populate the predictions (inputs for testing on the meta learner)
meta_X_test = pd.DataFrame({"cat_prob": cat_pred_test,"xgb_prob": xgb_pred_test,"logreg_prob": lr_pred_test,"lgbm_prob": lgbm_pred_test})
meta_y_test = y_test

# Predict probability using the meta learner
meta_pred_prob = final_meta_learner.predict_proba(meta_X_test)[:, 1]

# Evaluate and print results of testing data
class1_proportion_test = y_test.value_counts(normalize=True)[1]
test_metrics = evaluate_model(y_test, meta_pred_prob, 'Stack Ensemble', class1_proportion_test)

# Saving evaluation metrics
optimal_thresh, p, r, f, stability = test_metrics['threshold'], test_metrics['precision'], test_metrics['recall'], test_metrics['f1'], test_metrics['stability']

# Populate final stack ensemble predictions based on probabilities
final_pred = (meta_pred_prob > optimal_thresh).astype(int)

final_results = pd.DataFrame(data=[['Stack Ensemble', optimal_thresh, p, r, f, stability, confusion_matrix(y_test, final_pred)]], columns=['Model','Optimal Threshold', 'Precision', 'Recall', 'F1', 'Stability','Confusion Matrix'])

# Printing metrics
print(final_results.round(3).to_string(max_colwidth=50))
