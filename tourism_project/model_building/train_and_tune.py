"""
train_and_tune.py
- Loads train.csv (from tourism_project/data/)
- Builds preprocessing pipeline
- Runs Optuna hyperparameter search over several algorithms with MLflow logging
- Fits best model on full train set and writes best_pipeline.joblib and best_metrics.json to artifacts dir
"""
import os
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import mlflow
from optuna_integration.mlflow import MLflowCallback
import optuna

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import inspect

# Paths and constants
BASE = Path("tourism_project")
DATA_DIR = BASE / "data"
ARTIFACTS = BASE / "model_building" / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_DIR / "train.csv"
BEST_PIPE = ARTIFACTS / "best_pipeline.joblib"
METRICS_JSON = ARTIFACTS / "best_metrics.json"
OPTUNA_STUDY = ARTIFACTS / "optuna_study.pkl"

MLFLOW_DIR = (ARTIFACTS / "mlruns").resolve()
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri(str(MLFLOW_DIR))

# Helper for bagging compatibility
def make_bagging(base_estimator, n_estimators=10, max_samples=1.0, random_state=42, n_jobs=-1):
    sig = inspect.signature(BaggingClassifier.__init__)
    params = sig.parameters
    kwargs = {"n_estimators": int(n_estimators), "max_samples": max_samples, "random_state": random_state, "n_jobs": n_jobs}
    if "estimator" in params:
        kwargs["estimator"] = base_estimator
    elif "base_estimator" in params:
        kwargs["base_estimator"] = base_estimator
    else:
        try:
            return BaggingClassifier(estimator=base_estimator, **kwargs)
        except TypeError:
            return BaggingClassifier(base_estimator=base_estimator, **kwargs)
    return BaggingClassifier(**kwargs)

def load_data():
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"{TRAIN_CSV} not found. Run data_prep first.")
    df = pd.read_csv(TRAIN_CSV)
    return df

def build_preprocessor(df):
    target_col = "ProdTaken"
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]
    categorical_cols = [c for c in df.columns if c not in numeric_cols + [target_col]]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ], sparse_threshold=0)
    return preprocessor, numeric_cols, categorical_cols

def objective_factory(X, y, preprocessor):
    def objective(trial):
        algo = trial.suggest_categorical("algorithm", ["xgboost", "random_forest", "gradient_boosting", "adaboost", "bagging", "decision_tree"])
        if algo == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("xgb_n_estimators", 50, 200),
                "max_depth": trial.suggest_int("xgb_max_depth", 3, 8),
                "learning_rate": trial.suggest_float("xgb_lr", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("xgb_cbt", 0.6, 1.0),
                "use_label_encoder": False,
                "eval_metric": "logloss",
                "random_state": 42,
                "n_jobs": -1
            }
            model = XGBClassifier(**params)
        elif algo == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("rf_n_estimators", 50, 200),
                "max_depth": trial.suggest_int("rf_max_depth", 3, 30),
                "min_samples_split": trial.suggest_int("rf_min_split", 2, 8),
                "class_weight": "balanced"
            }
            model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        elif algo == "gradient_boosting":
            params = {
                "n_estimators": trial.suggest_int("gb_n_estimators", 50, 200),
                "learning_rate": trial.suggest_float("gb_lr", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("gb_max_depth", 3, 8)
            }
            model = GradientBoostingClassifier(**params, random_state=42)
        elif algo == "adaboost":
            params = {
                "n_estimators": trial.suggest_int("ada_n_estimators", 50, 200),
                "learning_rate": trial.suggest_float("ada_lr", 0.01, 1.0, log=True)
            }
            model = AdaBoostClassifier(**params, random_state=42)
        elif algo == "bagging":
            params = {
                "n_estimators": trial.suggest_int("bag_n_estimators", 10, 100),
                "max_samples": trial.suggest_float("bag_max_samples", 0.5, 1.0),
                "bag_dt_max_depth": trial.suggest_int("bag_dt_max_depth", 3, 8)
            }
            base = DecisionTreeClassifier(max_depth=int(params["bag_dt_max_depth"]), random_state=42)
            model = make_bagging(base_estimator=base, n_estimators=int(params["n_estimators"]), max_samples=float(params["max_samples"]), random_state=42, n_jobs=-1)
        else:
            params = {
                "max_depth": trial.suggest_int("dt_max_depth", 3, 30),
                "min_samples_split": trial.suggest_int("dt_min_split", 2, 10),
                "class_weight": "balanced"
            }
            model = DecisionTreeClassifier(**params, random_state=42)

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        return float(np.mean(scores))
    return objective

def main():
    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN not found in environment.")

    df = load_data()
    preprocessor, num_cols, cat_cols = build_preprocessor(df)
    X = df.drop(columns=["ProdTaken"])
    y = df["ProdTaken"].values

    # Optuna + MLflow setup
    mlflow_cb = MLflowCallback(tracking_uri=mlflow.get_tracking_uri(), metric_name="roc_auc")
    study = optuna.create_study(direction="maximize", study_name="wellness_tourism_multi_algo")
    objective = objective_factory(X, y, preprocessor)

    # run optimization
    n_trials = int(os.getenv("OPTUNA_TRIALS", "20"))
    print("Starting Optuna study with n_trials =", n_trials)
    try:
        study.optimize(objective, n_trials=n_trials, callbacks=[mlflow_cb], n_jobs=1)
    except Exception as e:
        print("Optuna with MLflow callback failed:", e)
        print("Falling back to run without MLflow callback.")
        study = optuna.create_study(direction="maximize", study_name="wellness_tourism_multi_algo_no_mlflow")
        study.optimize(objective, n_trials=n_trials, n_jobs=1)

    # Save study
    joblib.dump(study, OPTUNA_STUDY)
    print("Saved Optuna study to:", OPTUNA_STUDY)

    # Reconstruct best pipeline
    best = study.best_trial
    best_params = best.params
    algo = best_params.get("algorithm")

    # Rebuild model from best params (simple mapping)
    def build_model_from_params(params):
        algo = params["algorithm"]
        if algo == "xgboost":
            return XGBClassifier(n_estimators=int(params["xgb_n_estimators"]), max_depth=int(params["xgb_max_depth"]), learning_rate=float(params["xgb_lr"]), subsample=float(params["xgb_subsample"]), colsample_bytree=float(params["xgb_cbt"]), use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=-1)
        if algo == "random_forest":
            return RandomForestClassifier(n_estimators=int(params["rf_n_estimators"]), max_depth=int(params["rf_max_depth"]), min_samples_split=int(params["rf_min_split"]), class_weight="balanced", random_state=42, n_jobs=-1)
        if algo == "gradient_boosting":
            return GradientBoostingClassifier(n_estimators=int(params["gb_n_estimators"]), learning_rate=float(params["gb_lr"]), max_depth=int(params["gb_max_depth"]), random_state=42)
        if algo == "adaboost":
            return AdaBoostClassifier(n_estimators=int(params["ada_n_estimators"]), learning_rate=float(params["ada_lr"]), random_state=42)
        if algo == "bagging":
            base = DecisionTreeClassifier(max_depth=int(params["bag_dt_max_depth"]), random_state=42)
            return make_bagging(base_estimator=base, n_estimators=int(params["bag_n_estimators"]), max_samples=float(params["bag_max_samples"]), random_state=42, n_jobs=-1)
        return DecisionTreeClassifier(max_depth=int(params.get("dt_max_depth", 5)), min_samples_split=int(params.get("dt_min_split", 2)), class_weight="balanced", random_state=42)

    best_model = build_model_from_params(best_params)
    best_pipeline = Pipeline([("preprocessor", preprocessor), ("model", best_model)])

    # Fit best pipeline on full train
    best_pipeline.fit(X, y)
    joblib.dump(best_pipeline, BEST_PIPE)
    print("Saved best pipeline to:", BEST_PIPE)

    # Evaluate on holdout test set if available
    TEST_CSV = DATA_DIR / "test.csv"
    if TEST_CSV.exists():
        test_df = pd.read_csv(TEST_CSV)
        X_test = test_df.drop(columns=["ProdTaken"])
        y_test = test_df["ProdTaken"].values
        y_pred = best_pipeline.predict(X_test)
        y_proba = best_pipeline.predict_proba(X_test)[:,1] if hasattr(best_pipeline.named_steps["model"], "predict_proba") else None

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)) if y_proba is not None else None,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }
        with open(METRICS_JSON, "w") as fh:
            json.dump(metrics, fh, indent=2)
        print("Saved evaluation metrics to:", METRICS_JSON)
    else:
        print("Test CSV not found; skipping test evaluation.")

if __name__ == "__main__":
    main()
