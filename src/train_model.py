import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import os
import time

def train_and_evaluate(data_path, models_dir, target_col, strategy='fast'):
    """
    Trains models with dynamic target column, evaluating them and selecting the best.
    Strategies: 'fast' (Single RF), 'automl' (GridSearch across models)
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
        
    # Advanced EDA / Data Quality Extraction
    data_quality = {
        "missing_values": df.isnull().sum().to_dict(),
        "total_rows": len(df),
        "total_cols": len(df.columns)
    }
    
    # Preprocessing
    # Drop columns that have all unique values (like IDs)
    for col in df.columns:
        if df[col].nunique() == len(df) and col != target_col:
            df = df.drop([col], axis=1)
            
    # Attempt to convert object columns to numeric if possible (e.g. TotalCharges with spaces)
    for col in df.columns:
        if df[col].dtype == 'object' and col != target_col:
            # Check if majority of values are numeric-like
            numeric_vals = pd.to_numeric(df[col], errors='coerce')
            if numeric_vals.notna().mean() > 0.5:
                df[col] = numeric_vals
                
    # Define features
    X = df.drop([target_col], axis=1)
    y = df[target_col]
    
    # Impute missing values
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if len(numerical_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')
        X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
    else:
        num_imputer = None
        
    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
    else:
        cat_imputer = None
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        
    # Encode target variable
    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(y.astype(str))
    label_encoders['__TARGET__'] = y_encoder
    
    # Check if binary classification
    is_binary = len(np.unique(y)) == 2
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale numerical features
    scaler = StandardScaler()
    if len(numerical_cols) > 0:
        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # AutoML Logic
    print(f"Starting Training with strategy: {strategy}")
    models_to_try = []
    
    if strategy == 'fast':
        models_to_try = [
            ("Random Forest (Fast)", RandomForestClassifier(n_estimators=50, random_state=42), {})
        ]
    else:
        # AutoML mode
        models_to_try = [
            ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42), {'C': [0.1, 1.0]}),
            ("Random Forest", RandomForestClassifier(random_state=42), {'n_estimators': [50, 100], 'max_depth': [None, 10]}),
            ("Gradient Boosting", GradientBoostingClassifier(random_state=42), {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]})
        ]
        
    best_model = None
    best_name = ""
    best_score = -1
    leaderboard = []
    
    for name, model, params in models_to_try:
        print(f"Training {name}...")
        start_time = time.time()
        
        if strategy == 'automl' and len(params) > 0:
            clf = GridSearchCV(model, params, cv=3, scoring='accuracy', n_jobs=-1)
            clf.fit(X_train, y_train)
            current_model = clf.best_estimator_
        else:
            current_model = model
            current_model.fit(X_train, y_train)
            
        y_pred = current_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        train_time = time.time() - start_time
        
        leaderboard.append({"model": name, "accuracy": acc, "time_sec": round(train_time, 2)})
        
        if acc > best_score:
            best_score = acc
            best_model = current_model
            best_name = name

    print(f"Best Model Selected: {best_name} with Accuracy {best_score:.4f}")
    
    # Advanced Metrics for Best Model
    y_pred = best_model.predict(X_test)
    
    metrics = {
        "best_model_name": best_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "f1": f1_score(y_test, y_pred, average='weighted', zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classes": [str(c) for c in y_encoder.classes_],
        "leaderboard": sorted(leaderboard, key=lambda x: x['accuracy'], reverse=True)
    }
    
    if is_binary and hasattr(best_model, "predict_proba"):
        y_prob = best_model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
    else:
        metrics["roc_auc"] = "N/A"
        
    # Feature Importance (if supported)
    features = X.columns.tolist()
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        fi_dict = {f: float(i) for f, i in zip(features, importances)}
    elif hasattr(best_model, "coef_"):
        importances = np.abs(best_model.coef_[0])
        fi_dict = {f: float(i) for f, i in zip(features, importances)}
    else:
        fi_dict = {f: 1.0 / len(features) for f in features} # Dummy uniform importance
        
    # Generate schema for frontend
    schema = []
    for col in features:
        col_type = "categorical" if col in categorical_cols else "numerical"
        options = []
        if col_type == "categorical":
            options = list(label_encoders[col].classes_)
        schema.append({
            "name": col,
            "type": col_type,
            "options": [str(o) for o in options]
        })
        
    # Save artifacts
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(best_model, os.path.join(models_dir, 'best_model.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    joblib.dump(num_imputer, os.path.join(models_dir, 'num_imputer.pkl'))
    joblib.dump(cat_imputer, os.path.join(models_dir, 'cat_imputer.pkl'))
    joblib.dump(label_encoders, os.path.join(models_dir, 'label_encoders.pkl'))
    joblib.dump(features, os.path.join(models_dir, 'feature_names.pkl'))
    joblib.dump(numerical_cols, os.path.join(models_dir, 'numerical_cols.pkl'))
    joblib.dump(categorical_cols, os.path.join(models_dir, 'categorical_cols.pkl'))
    joblib.dump(schema, os.path.join(models_dir, 'schema.pkl'))
    joblib.dump(target_col, os.path.join(models_dir, 'target_col.pkl'))
    joblib.dump(fi_dict, os.path.join(models_dir, 'feature_importances.pkl'))
    joblib.dump(metrics, os.path.join(models_dir, 'metrics.pkl'))
    joblib.dump(data_quality, os.path.join(models_dir, 'data_quality.pkl'))
    
    # Compute basic correlations for EDA (only on numerical cols for simplicity)
    if len(numerical_cols) > 1:
        corr_matrix = df[numerical_cols].corr().fillna(0).to_dict()
    else:
        corr_matrix = {}
    joblib.dump(corr_matrix, os.path.join(models_dir, 'correlation_matrix.pkl'))

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'customer_churn.csv')
    models_dir = os.path.join(base_dir, 'models')
    
    # Default initial run
    train_and_evaluate(data_path, models_dir, 'Churn', strategy='fast')
