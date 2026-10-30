#!/usr/bin/env python3
"""
ML KTAS Classification with Cross-Fold Validation and Hyperparameter Tuning

This script performs KTAS (Korean Triage and Acuity Scale) classification on Kaggle data using:
- Logistic Regression
- XGBoost
- MLP (simple)
- BioBERT + MLP (with text embeddings)

With K-Fold cross-validation and hyperparameter grid search.
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import argparse
from datetime import datetime

# Add parent directory to path for utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils.utils as utils

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score, cohen_kappa_score


def load_and_preprocess_data(train_path: str, test_path: str):
    """
    Load and preprocess the KTAS data (structured features only).
    
    Args:
        train_path: Path to training CSV
        test_path: Path to test CSV
        
    Returns:
        X_train, y_train, X_test, y_test: Preprocessed data splits
    """
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # List of vital signs columns
    vital_signs_cols = ['NRS_pain', 'SBP', 'DBP', 'HR', 'RR', 'BT']
    
    # Normalize vital signs using Min-Max scaling for train data
    scaler = MinMaxScaler()
    train_df[vital_signs_cols] = scaler.fit_transform(train_df[vital_signs_cols].fillna(train_df[vital_signs_cols].mean()))
    test_df[vital_signs_cols] = scaler.transform(test_df[vital_signs_cols].fillna(test_df[vital_signs_cols].mean()))
    
    # Drop non-feature columns
    cols_to_drop = ['Length of stay_min', 'Group', 'Chief_complain', 'Diagnosis in ED', 'KTAS_RN']
    train_df = train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns])
    test_df = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns])
    
    # One-hot encode categorical variables
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    categorical_cols = ['Sex', 'Arrival mode', 'Mental']
    categorical_cols = [c for c in categorical_cols if c in train_df.columns]
    
    if categorical_cols:
        train_categories = train_df[categorical_cols]
        test_categories = test_df[categorical_cols]
        
        encoder.fit(train_categories)
        
        train_encoded = encoder.transform(train_categories)
        test_encoded = encoder.transform(test_categories)
        
        encoded_columns = encoder.get_feature_names_out(categorical_cols)
        
        train_encoded_df = pd.DataFrame(train_encoded, columns=encoded_columns, index=train_df.index)
        test_encoded_df = pd.DataFrame(test_encoded, columns=encoded_columns, index=test_df.index)
        
        train_df = train_df.drop(columns=categorical_cols).join(train_encoded_df)
        test_df = test_df.drop(columns=categorical_cols).join(test_encoded_df)
    
    # Scale Age separately since it's numeric
    if 'Age' in train_df.columns:
        age_scaler = MinMaxScaler()
        train_df['Age'] = age_scaler.fit_transform(train_df[['Age']])
        test_df['Age'] = age_scaler.transform(test_df[['Age']])
    
    # Drop additional columns if present
    extra_cols_to_drop = ['Patients number per hour', 'Injury', 'Pain']
    train_df = train_df.drop(columns=[c for c in extra_cols_to_drop if c in train_df.columns])
    test_df = test_df.drop(columns=[c for c in extra_cols_to_drop if c in test_df.columns])
    
    # Fill any remaining NAs with mean
    numeric_cols = train_df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        if col != 'KTAS_expert':
            train_df[col] = train_df[col].fillna(train_df[col].mean())
            test_df[col] = test_df[col].fillna(test_df[col].mean())
    
    # Define target column
    target_col = "KTAS_expert"
    features = [col for col in train_df.columns if col != target_col]
    
    X_train = train_df[features]
    y_train = train_df[target_col].values - 1  # Adjust class labels to start from 0
    
    X_test = test_df[features]
    y_test = test_df[target_col].values - 1  # Adjust class labels to start from 0
    
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    return X_train, y_train, X_test, y_test


def load_and_preprocess_data_with_embeddings(train_path: str, test_path: str, data_dir: str, 
                                              pca_components: int = 50, recompute_embeddings: bool = False):
    """
    Load and preprocess the KTAS data with BioBERT embeddings for text fields.
    
    Args:
        train_path: Path to training CSV
        test_path: Path to test CSV
        data_dir: Directory containing embedding files
        pca_components: Number of PCA components for dimensionality reduction
        recompute_embeddings: Whether to recompute embeddings even if cached files exist
        
    Returns:
        X_train, y_train, X_test, y_test: Preprocessed data with embeddings
    """
    print("Loading data with BioBERT embeddings...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Drop columns we don't need for features
    cols_to_drop = ['Length of stay_min', 'Group', 'KTAS_RN']
    train_df = train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns])
    test_df = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns])
    
    # Embedding file paths
    train_chief_emb_path = os.path.join(data_dir, 'KTAS_train_chiefcomplaint_embeddings.npy')
    test_chief_emb_path = os.path.join(data_dir, 'KTAS_test_chiefcomplaint_embeddings.npy')
    train_diag_emb_path = os.path.join(data_dir, 'KTAS_train_diagnosis_embeddings.npy')
    test_diag_emb_path = os.path.join(data_dir, 'KTAS_test_diagnosis_embeddings.npy')
    
    # Check if embeddings exist, otherwise compute them
    embeddings_exist = all(os.path.exists(p) for p in [train_chief_emb_path, test_chief_emb_path, 
                                                        train_diag_emb_path, test_diag_emb_path])
    
    if not embeddings_exist or recompute_embeddings:
        print("Computing BioBERT embeddings (this may take a while)...")
        from sentence_transformers import SentenceTransformer
        
        from utils.predictors import resolve_sentence_encoder_name, format_sentence_encoder_load_error

        model_name = resolve_sentence_encoder_name('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')
        try:
            embedding_model = SentenceTransformer(model_name)
        except Exception as exc:
            raise RuntimeError(format_sentence_encoder_load_error(model_name, exc)) from exc
        
        # Chief complaint embeddings
        print("  Computing chief complaint embeddings...")
        train_chief_embeddings = embedding_model.encode(
            train_df['Chief_complain'].fillna('').tolist(), 
            show_progress_bar=True
        )
        test_chief_embeddings = embedding_model.encode(
            test_df['Chief_complain'].fillna('').tolist(), 
            show_progress_bar=True
        )
        np.save(train_chief_emb_path, train_chief_embeddings)
        np.save(test_chief_emb_path, test_chief_embeddings)
        
        # Diagnosis embeddings
        print("  Computing diagnosis embeddings...")
        train_diag_embeddings = embedding_model.encode(
            train_df['Diagnosis in ED'].fillna('').tolist(), 
            show_progress_bar=True
        )
        test_diag_embeddings = embedding_model.encode(
            test_df['Diagnosis in ED'].fillna('').tolist(), 
            show_progress_bar=True
        )
        np.save(train_diag_emb_path, train_diag_embeddings)
        np.save(test_diag_emb_path, test_diag_embeddings)
    else:
        print("  Loading cached embeddings...")
        train_chief_embeddings = np.load(train_chief_emb_path)
        test_chief_embeddings = np.load(test_chief_emb_path)
        train_diag_embeddings = np.load(train_diag_emb_path)
        test_diag_embeddings = np.load(test_diag_emb_path)
    
    # Apply PCA to reduce dimensionality
    if pca_components > 0:
        print(f"  Applying PCA (n_components={pca_components})...")
        pca_chief = PCA(n_components=pca_components)
        train_chief_pca = pca_chief.fit_transform(train_chief_embeddings)
        test_chief_pca = pca_chief.transform(test_chief_embeddings)
        
        pca_diag = PCA(n_components=pca_components)
        train_diag_pca = pca_diag.fit_transform(train_diag_embeddings)
        test_diag_pca = pca_diag.transform(test_diag_embeddings)
    else:
        train_chief_pca = train_chief_embeddings
        test_chief_pca = test_chief_embeddings
        train_diag_pca = train_diag_embeddings
        test_diag_pca = test_diag_embeddings
    
    # Drop text columns now that we have embeddings
    train_df = train_df.drop(columns=['Chief_complain', 'Diagnosis in ED'], errors='ignore')
    test_df = test_df.drop(columns=['Chief_complain', 'Diagnosis in ED'], errors='ignore')
    
    # Fill NAs in numeric columns
    numeric_cols = train_df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        if col != 'KTAS_expert':
            train_df[col] = train_df[col].fillna(train_df[col].mean())
            test_df[col] = test_df[col].fillna(test_df[col].mean())
    
    # Define target column
    target_col = "KTAS_expert"
    features = [col for col in train_df.columns if col != target_col]
    
    X_train_structured = train_df[features].values
    y_train = train_df[target_col].values - 1  # Adjust class labels to start from 0
    
    X_test_structured = test_df[features].values
    y_test = test_df[target_col].values - 1
    
    # Combine structured features with embeddings
    X_train = np.hstack([X_train_structured, train_chief_pca, train_diag_pca])
    X_test = np.hstack([X_test_structured, test_chief_pca, test_diag_pca])
    
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  - Structured features: {X_train_structured.shape[1]}")
    print(f"  - Chief complaint PCA: {train_chief_pca.shape[1]}")
    print(f"  - Diagnosis PCA: {train_diag_pca.shape[1]}")
    print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    return X_train, y_train, X_test, y_test


def get_logistic_regression_params():
    """Define hyperparameter grid for Logistic Regression."""
    return [
        # Grid 1: Solvers that only support L2
        {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l2'],
            'solver': ['lbfgs'],
            'max_iter': [500, 1000, 2000, 4000],
            'class_weight': [None, 'balanced']
        },
        # Grid 2: Solvers that support L1 and L2
        {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2'],
            'solver': ['saga'],
            'max_iter': [1000, 2000, 4000], # Saga often needs higher max_iter
            'class_weight': [None, 'balanced']
        }
    ]


def get_xgboost_params():
    """Define hyperparameter grid for XGBoost."""
    return {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'min_child_weight': [1, 3],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }


def get_mlp_params():
    """Define hyperparameter grid for MLP."""
    return {
        'hidden_layer_sizes': [(100, 100), (200, 200)],
        'activation': ['relu'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        'learning_rate_init': [0.001, 0.01],
        'batch_size': [32, 128]
    }


def get_biobert_mlp_params():
    """Define hyperparameter grid for BioBERT + MLP."""
    return {
        'hidden_layer_sizes': [(100, 100), (200, 200), (512, 256)],
        'activation': ['relu'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        'learning_rate_init': [0.001, 0.01],
        'batch_size': [32, 128]
    }


# PCA components to try as hyperparameter
PCA_COMPONENTS_GRID = [50, 100, 200]


def run_cross_validation(X, y, model, param_grid, cv_folds=5, n_jobs=-1):
    """
    Run cross-validation with hyperparameter tuning.
    
    Args:
        X: Feature matrix
        y: Target vector
        model: Base model to tune
        param_grid: Dictionary of hyperparameters to search
        cv_folds: Number of cross-validation folds
        n_jobs: Number of parallel jobs
        
    Returns:
        grid_search: Fitted GridSearchCV object
    """
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=n_jobs,
        verbose=2,
        return_train_score=True,
        refit=True  # Retrain on all training data with best params
    )
    
    grid_search.fit(X, y)
    
    return grid_search


def compute_triage_metrics(y_true, y_pred):
    """
    Compute comprehensive triage metrics including under/over-triage rates.
    
    Under-triage: predicted acuity > true acuity (patient assigned lower priority than needed)
    Over-triage: predicted acuity < true acuity (patient assigned higher priority than needed)
    
    Args:
        y_true: True acuity labels
        y_pred: Predicted acuity labels
        
    Returns:
        metrics: Dictionary containing all metrics
    """
    # Get standard metrics from utils
    base_metrics = utils.evaluate_predictions(y_pred, y_true, ordinal=True, by_class=True)
    
    # Compute under-triage and over-triage rates
    n_samples = len(y_true)
    under_triage = np.sum(y_pred > y_true)  # Higher number = lower priority
    over_triage = np.sum(y_pred < y_true)   # Lower number = higher priority
    
    under_triage_rate = under_triage / n_samples * 100  # as percentage
    over_triage_rate = over_triage / n_samples * 100    # as percentage
    
    # Add triage-specific metrics
    base_metrics['overall']['under_triage_rate'] = under_triage_rate
    base_metrics['overall']['over_triage_rate'] = over_triage_rate
    base_metrics['overall']['under_triage_count'] = int(under_triage)
    base_metrics['overall']['over_triage_count'] = int(over_triage)
    
    return base_metrics


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        
    Returns:
        metrics: Dictionary of evaluation metrics
        y_pred: Predictions
    """
    y_pred = model.predict(X_test)
    metrics = compute_triage_metrics(y_test, y_pred)
    return metrics, y_pred


def save_results(results: dict, output_dir: str, model_name: str):
    """Save results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"Triage-KTAS_{model_name}_CV_{timestamp}.json")
    
    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    results = convert_types(results)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    return output_file


def print_metrics(metrics: dict, model_name: str):
    """Print formatted metrics for a model."""
    print(f"\n{model_name} Test set metrics:")
    print(f"  Accuracy: {metrics['overall']['accuracy']:.4f}")
    print(f"  F1 Score: {metrics['overall']['f1_score']:.4f}")
    print(f"  Quadratic Kappa: {metrics['overall']['quadratic_kappa']:.4f}")
    print(f"  MAE: {metrics['overall']['mae']:.4f}")
    print(f"  MSE: {metrics['overall']['mse']:.4f}")
    print(f"  Under-triage: {metrics['overall']['under_triage_rate']:.2f}%")
    print(f"  Over-triage: {metrics['overall']['over_triage_rate']:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='ML KTAS Classification with Cross-Validation')
    parser.add_argument('--train-path', type=str, 
                        default='../data/kaggle/train_numeric.csv',
                        help='Path to training data')
    parser.add_argument('--test-path', type=str,
                        default='../data/kaggle/test_numeric.csv',
                        help='Path to test data')
    parser.add_argument('--data-dir', type=str, default='../data/kaggle',
                        help='Directory containing embedding files (for BioBERT+MLP)')
    parser.add_argument('--output-dir', type=str, default='../results/Triage-KTAS',
                        help='Output directory for results')
    parser.add_argument('--cv-folds', type=int, default=4,
                        help='Number of cross-validation folds')
    parser.add_argument('--model', type=str, choices=['logistic', 'xgboost', 'mlp', 'biobert_mlp', 'all'], 
                        default='all', help='Model to train (biobert_mlp uses text embeddings)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Number of parallel jobs (-1 for all CPUs)')
    parser.add_argument('--recompute-embeddings', action='store_true',
                        help='Recompute BioBERT embeddings even if cached')
    
    args = parser.parse_args()
    
    # Resolve paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(script_dir, args.train_path)
    test_path = os.path.join(script_dir, args.test_path)
    data_dir = os.path.join(script_dir, args.data_dir)
    output_dir = os.path.join(script_dir, args.output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data (for non-embedding models)
    X_train, y_train, X_test, y_test = load_and_preprocess_data(train_path, test_path)
    
    all_results = {}
    
    # Logistic Regression
    if args.model in ['logistic', 'all']:
        print("\n" + "="*60)
        print("Training Logistic Regression with Cross-Validation")
        print("="*60)
        
        lr_params = get_logistic_regression_params()
        lr_model = LogisticRegression(random_state=42, n_jobs=args.n_jobs)
        lr_grid = run_cross_validation(
            X_train, y_train, lr_model, lr_params,
            cv_folds=args.cv_folds, n_jobs=args.n_jobs
        )
        
        print(f"\nBest parameters: {lr_grid.best_params_}")
        print(f"Best CV score: {lr_grid.best_score_:.4f}")
        
        # Evaluate on test set
        lr_metrics, lr_pred = evaluate_model(lr_grid.best_estimator_, X_test, y_test)
        print_metrics(lr_metrics, "Logistic Regression")
        
        all_results['logistic_regression'] = {
            'best_params': lr_grid.best_params_,
            'best_cv_score': lr_grid.best_score_,
            'cv_results': {
                'mean_test_score': lr_grid.cv_results_['mean_test_score'].tolist(),
                'std_test_score': lr_grid.cv_results_['std_test_score'].tolist(),
                'params': [str(p) for p in lr_grid.cv_results_['params']]
            },
            'test_metrics': lr_metrics
        }
        
        # Save predictions
        pd.DataFrame({'prediction': lr_pred + 1}).to_csv(
            os.path.join(output_dir, 'Triage-KTAS_LogisticRegression_CV.csv'), 
            index=False
        )
    
    # XGBoost
    if args.model in ['xgboost', 'all']:
        print("\n" + "="*60)
        print("Training XGBoost with Cross-Validation")
        print("="*60)
        
        xgb_params = get_xgboost_params()
        xgb_model = XGBClassifier(
            random_state=42,
            n_jobs=args.n_jobs,
            eval_metric='mlogloss'
        )
        
        xgb_grid = run_cross_validation(
            X_train, y_train, xgb_model, xgb_params,
            cv_folds=args.cv_folds, n_jobs=args.n_jobs
        )
        
        print(f"\nBest parameters: {xgb_grid.best_params_}")
        print(f"Best CV score: {xgb_grid.best_score_:.4f}")
        
        # Evaluate on test set
        xgb_metrics, xgb_pred = evaluate_model(xgb_grid.best_estimator_, X_test, y_test)
        print_metrics(xgb_metrics, "XGBoost")
        
        all_results['xgboost'] = {
            'best_params': xgb_grid.best_params_,
            'best_cv_score': xgb_grid.best_score_,
            'cv_results': {
                'mean_test_score': xgb_grid.cv_results_['mean_test_score'].tolist(),
                'std_test_score': xgb_grid.cv_results_['std_test_score'].tolist(),
                'params': [str(p) for p in xgb_grid.cv_results_['params']]
            },
            'test_metrics': xgb_metrics
        }
        
        # Save predictions
        pd.DataFrame({'prediction': xgb_pred + 1}).to_csv(
            os.path.join(output_dir, 'Triage-KTAS_XGBoost_CV.csv'), 
            index=False
        )
    
    # MLP
    if args.model in ['mlp', 'all']:
        print("\n" + "="*60)
        print("Training MLP with Cross-Validation")
        print("="*60)
        
        mlp_params = get_mlp_params()
        mlp_model = MLPClassifier(
            random_state=42,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        mlp_grid = run_cross_validation(
            X_train, y_train, mlp_model, mlp_params,
            cv_folds=args.cv_folds, n_jobs=args.n_jobs
        )
        
        print(f"\nBest parameters: {mlp_grid.best_params_}")
        print(f"Best CV score: {mlp_grid.best_score_:.4f}")
        
        # Evaluate on test set
        mlp_metrics, mlp_pred = evaluate_model(mlp_grid.best_estimator_, X_test, y_test)
        print_metrics(mlp_metrics, "MLP")
        
        all_results['mlp'] = {
            'best_params': {k: str(v) if isinstance(v, tuple) else v for k, v in mlp_grid.best_params_.items()},
            'best_cv_score': mlp_grid.best_score_,
            'cv_results': {
                'mean_test_score': mlp_grid.cv_results_['mean_test_score'].tolist(),
                'std_test_score': mlp_grid.cv_results_['std_test_score'].tolist(),
                'params': [str(p) for p in mlp_grid.cv_results_['params']]
            },
            'test_metrics': mlp_metrics
        }
        
        # Save predictions
        pd.DataFrame({'prediction': mlp_pred + 1}).to_csv(
            os.path.join(output_dir, 'Triage-KTAS_MLP_CV.csv'), 
            index=False
        )
    
    # BioBERT + MLP (search over PCA components as hyperparameter)
    if args.model in ['biobert_mlp', 'all']:
        print("\n" + "="*60)
        print("Training BioBERT + MLP with Cross-Validation")
        print("="*60)
        
        best_biobert_score = -1
        best_biobert_results = None
        best_pca_components = None
        
        for pca_components in PCA_COMPONENTS_GRID:
            print(f"\n--- PCA Components: {pca_components} ---")
            
            # Load data with embeddings
            X_train_emb, y_train_emb, X_test_emb, y_test_emb = load_and_preprocess_data_with_embeddings(
                train_path, test_path, data_dir,
                pca_components=pca_components,
                recompute_embeddings=args.recompute_embeddings
            )
            
            biobert_mlp_params = get_biobert_mlp_params()
            biobert_mlp_model = MLPClassifier(
                random_state=42,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1
            )
            
            biobert_mlp_grid = run_cross_validation(
                X_train_emb, y_train_emb, biobert_mlp_model, biobert_mlp_params,
                cv_folds=args.cv_folds, n_jobs=args.n_jobs
            )
            
            print(f"Best parameters: {biobert_mlp_grid.best_params_}")
            print(f"Best CV score: {biobert_mlp_grid.best_score_:.4f}")
            
            if biobert_mlp_grid.best_score_ > best_biobert_score:
                best_biobert_score = biobert_mlp_grid.best_score_
                best_pca_components = pca_components
                best_biobert_results = {
                    'grid': biobert_mlp_grid,
                    'X_test': X_test_emb,
                    'y_test': y_test_emb
                }
        
        print(f"\n=== Best PCA Components: {best_pca_components} ===")
        
        # Evaluate on test set with best configuration
        biobert_metrics, biobert_pred = evaluate_model(
            best_biobert_results['grid'].best_estimator_, 
            best_biobert_results['X_test'], 
            best_biobert_results['y_test']
        )
        print_metrics(biobert_metrics, "BioBERT + MLP")
        
        all_results['biobert_mlp'] = {
            'best_params': {k: str(v) if isinstance(v, tuple) else v 
                          for k, v in best_biobert_results['grid'].best_params_.items()},
            'best_cv_score': best_biobert_results['grid'].best_score_,
            'best_pca_components': best_pca_components,
            'pca_components_searched': PCA_COMPONENTS_GRID,
            'cv_results': {
                'mean_test_score': best_biobert_results['grid'].cv_results_['mean_test_score'].tolist(),
                'std_test_score': best_biobert_results['grid'].cv_results_['std_test_score'].tolist(),
                'params': [str(p) for p in best_biobert_results['grid'].cv_results_['params']]
            },
            'test_metrics': biobert_metrics
        }
        
        # Save predictions
        pd.DataFrame({'prediction': biobert_pred + 1}).to_csv(
            os.path.join(output_dir, 'Triage-KTAS_BioBERT_MLP_CV.csv'), 
            index=False
        )
    
    # Save all results
    save_results(all_results, output_dir, 'combined')
    
    print("\n" + "="*60)
    print("Cross-validation and hyperparameter tuning complete!")
    print("="*60)


if __name__ == "__main__":
    main()
