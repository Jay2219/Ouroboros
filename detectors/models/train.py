import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import os
import sys
import shap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from detectors.features.pipeline import extract_features, get_X_y
from eval.detection import evaluate_detection

def train_primary_classifier(events: list):
    """Train XGBoost on the given events (already generated + legitimate)."""
    df_features = extract_features(events)
    if df_features.empty:
        print("No features to train on.")
        return None
        
    X, y = get_X_y(df_features)
    
    # Needs at least one positive and one negative class
    if len(y.unique()) < 2:
        print("Dataset does not contain both classes. Cannot train.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    metrics = evaluate_detection(y_test.values, y_pred_proba, target_fpr=0.01)
    
    print("Evaluation Metrics (at 1% FPR target):")
    print(metrics)
    
    print("Generating SHAP Explainer...")
    explainer = shap.TreeExplainer(clf)
    
    return clf, explainer

def train_anomaly_detector(events: list):
    """
    Train Isolation Forest for zero-day / sparse positive attacks.
    Ideally trained primarily on legitimate data to learn the 'normal' distribution.
    """
    df_features = extract_features(events)
    if df_features.empty:
        return None
        
    X, y = get_X_y(df_features)
    
    # Train on all available data (IsolationForest is unsupervised, but we could filter to y==0)
    print("Training Isolation Forest...")
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_forest.fit(X)
    
    # Optional: basic evaluation of how many actual frauds it flagged as anomalies
    preds = iso_forest.predict(X) # Returns 1 for normal, -1 for anomaly
    anomaly_flags = (preds == -1).astype(int)
    
    from sklearn.metrics import classification_report
    if len(y.unique()) > 1:
        print("Anomaly Detector (Isolation Forest) Performance on Training Data:")
        print(classification_report(y, anomaly_flags))
    
    return iso_forest

if __name__ == "__main__":
    print("This module is meant to be called from a pipeline script, but can run isolated tests here.")
