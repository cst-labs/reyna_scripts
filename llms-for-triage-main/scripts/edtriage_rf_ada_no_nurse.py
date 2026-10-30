#!/usr/bin/env python3
"""
Random Forest / AdaBoost training for the KTAS "edtriage" dataset, replicating
the pipeline in edtriage.ipynb, for the feature set that excludes KTAS_RN
(the triage nurse's own KTAS assessment) as an input.

Pipeline (matches the notebook):
  - Load the pre-split KTAS train/test CSVs (same random_state=42 stratified
    split as the notebook's train_test_split on the original .xlsx).
  - Drop 'Mental' and 'KTAS_RN' (low variance / not usable as a real-time
    prediction input).
  - Clean + TF-IDF the Chief_complain text, reduce with TruncatedSVD.
  - Median-impute vitals, remove outliers (IsolationForest) on the train set.
  - Select the significant structured features (sig_feat), scale with
    RobustScaler.
  - Balance classes with SMOTE (on structured + text features combined).
  - RandomizedSearchCV hyperparameter tuning (cv=5, scoring=f1_weighted) for
    RandomForestClassifier and AdaBoostClassifier.
  - Evaluate the tuned models on the held-out test set: weighted F1,
    confusion matrix, classification report, weighted OvR ROC-AUC.

Deviation from the notebook: AdaBoostClassifier's `algorithm` hyperparameter
(SAMME / SAMME.R) was removed in scikit-learn >= 1.6, so it is dropped from
the AdaBoost parameter grid here.
"""

import argparse
import json
import os
import pickle
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import AdaBoostClassifier, IsolationForest, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import SMOTE

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Features selected in the notebook as significant + usable for real-time
# prediction (KTAS_RN, Disposition, Error_group, mistriage etc. excluded).
SIG_FEAT = ['Group', 'Sex', 'Age', 'Injury', 'Pain', 'NRS_pain', 'SBP', 'RR', 'BT', 'Saturation']

VITAL_NUMERIC_COLS = ['SBP', 'DBP', 'HR', 'RR', 'BT', 'Saturation']
IMPUTE_COLS = ['Saturation', 'NRS_pain', 'DBP', 'SBP', 'RR', 'HR', 'BT']
# 'KTAS duration_min' and 'Length of stay_min' are post-visit/outcome fields,
# not available at real-time prediction, so they are dropped entirely (not
# even used for outlier detection).
OUTLIER_FEATURES = ['BT', 'DBP', 'HR', 'Patients number per hour', 'RR', 'SBP', 'Saturation']

ABBR_DICT = {
    "f/c": "fever chills",
    "ruq": "right upper quadrant",
    "n/v": "nausea vomiting",
    "cp": "chest pain",
    "nos": "not otherwise specified",
    "loc": "loss of consciousness",
    "lbp": "lower back pain",
    "ant": "anterior",
    "avf": "arteriovenous fistula",
    "peg": "percutaneous endoscopic gastrostomy",
    "abd": "abdomen",
    "cpr": "cardiopulmonary resuscitation",
    "llq": "left lower quadrant",
    "fb": "foreign body",
    "ptbd": "percutaneous transhepatic biliary drainage",
    "a/": "associated with",
    "rlq": "right lower quadrant",
    "hb": "hemoglobin",
    "dz": "dizziness",
    "sx": "symptoms",
    "bsl": "blood sugar level",
    "pprom": "preterm premature rupture of membranes",
    "ptx": "pneumothorax",
    "c/s": "cesarean section",
    "g/w": "general weakness",
}

PARAM_GRID_RF = {
    'n_estimators': [50, 100],
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [2, 5],
    'max_features': ['sqrt', None],
    'bootstrap': [True, False],
}

# 'algorithm' (SAMME / SAMME.R) dropped: removed from AdaBoostClassifier in
# scikit-learn >= 1.6.
PARAM_GRID_ADA = {
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.05, 0.1],
    'estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=3)],
}


def ensure_nltk_data():
    for pkg, path in [('stopwords', 'corpora/stopwords'), ('wordnet', 'corpora/wordnet')]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg)


def expand_abbreviations(text):
    text = str(text).lower()
    for abbr, full_form in ABBR_DICT.items():
        text = text.replace(abbr, full_form)
    return text


def clean_text(text):
    text = expand_abbreviations(text)
    text = re.sub(r'[^a-z\s,-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def build_text_pipeline(train_df, test_df):
    ensure_nltk_data()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess(text):
        words = text.split()
        words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
        return ' '.join(words)

    train_clean = train_df['Chief_complain'].apply(clean_text).apply(preprocess)
    test_clean = test_df['Chief_complain'].apply(clean_text).apply(preprocess)

    tfidf_chief = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    text_train = tfidf_chief.fit_transform(train_clean).toarray()
    text_test = tfidf_chief.transform(test_clean).toarray()
    return text_train, text_test, tfidf_chief


def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df[VITAL_NUMERIC_COLS] = train_df[VITAL_NUMERIC_COLS].apply(pd.to_numeric, errors='coerce')
    test_df[VITAL_NUMERIC_COLS] = test_df[VITAL_NUMERIC_COLS].apply(pd.to_numeric, errors='coerce')

    # Drop the triage nurse's KTAS assessment (and 'Mental', dropped for low
    # variance in the notebook) -- this is the "without triage nurse" variant.
    # Also drop 'KTAS duration_min' and 'Length of stay_min': both are only
    # known after the visit, so they can't be real-time prediction inputs.
    drop_cols = ['Mental', 'KTAS_RN', 'KTAS duration_min', 'Length of stay_min']
    train_df = train_df.drop(columns=drop_cols)
    test_df = test_df.drop(columns=drop_cols)

    return train_df, test_df


def impute_missing(train_df, test_df):
    train_df[['Saturation', 'NRS_pain']] = train_df[['Saturation', 'NRS_pain']].apply(pd.to_numeric, errors='coerce')
    test_df[['Saturation', 'NRS_pain']] = test_df[['Saturation', 'NRS_pain']].apply(pd.to_numeric, errors='coerce')

    imputer = SimpleImputer(strategy='median')
    train_df[IMPUTE_COLS] = imputer.fit_transform(train_df[IMPUTE_COLS])
    test_df[IMPUTE_COLS] = imputer.transform(test_df[IMPUTE_COLS])
    return train_df, test_df


def remove_outliers(train_df, contamination=0.05, random_state=42):
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    outliers = iso_forest.fit_predict(train_df[OUTLIER_FEATURES]) == -1

    for col in OUTLIER_FEATURES:
        median_val = train_df.loc[~outliers, col].median()
        train_df.loc[outliers, col] = median_val
    return train_df


def build_features(train_df, test_df, text_train, text_test, random_state=42):
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    X_train = train_df[SIG_FEAT]
    y_train = train_df['KTAS_expert'].values
    X_test = test_df[SIG_FEAT]
    y_test = test_df['KTAS_expert'].values

    X_full_train = np.hstack([X_train.values, text_train])
    X_full_test = np.hstack([X_test.values, text_test])

    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_full_train, y_train)
    X_test_smote, y_test_smote = X_full_test, y_test

    n_struct = len(SIG_FEAT)
    struct_train, text_train_part = X_train_smote[:, :n_struct], X_train_smote[:, n_struct:]
    struct_test, text_test_part = X_test_smote[:, :n_struct], X_test_smote[:, n_struct:]

    scaler = RobustScaler()
    struct_train_scaled = scaler.fit_transform(struct_train)
    struct_test_scaled = scaler.transform(struct_test)

    svd = TruncatedSVD(n_components=50, random_state=random_state)
    text_train_reduced = svd.fit_transform(text_train_part)
    text_test_reduced = svd.transform(text_test_part)

    struct_train_df = pd.DataFrame(struct_train_scaled, columns=SIG_FEAT)
    struct_test_df = pd.DataFrame(struct_test_scaled, columns=SIG_FEAT)
    text_train_df = pd.DataFrame(text_train_reduced, columns=[f'svd_{i + 1}' for i in range(50)])
    text_test_df = pd.DataFrame(text_test_reduced, columns=[f'svd_{i + 1}' for i in range(50)])

    final_train_df = pd.concat([struct_train_df, text_train_df], axis=1)
    final_test_df = pd.concat([struct_test_df, text_test_df], axis=1)

    return final_train_df, final_test_df, y_train_smote, y_test_smote, scaler, svd


def random_searchcv(model, param_grid, n_iter=10, cv=5, n_jobs=-1, random_state=42):
    return RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=n_jobs,
        random_state=random_state,
    )


def evaluate_model(name, search, final_test_df, y_test_smote):
    best_estimator = search.best_estimator_
    y_pred = best_estimator.predict(final_test_df)
    y_proba = best_estimator.predict_proba(final_test_df)

    test_f1 = f1_score(y_test_smote, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test_smote, y_proba, multi_class='ovr', average='weighted')
    accuracy = accuracy_score(y_test_smote, y_pred)
    qwk = cohen_kappa_score(y_test_smote, y_pred, weights='quadratic')
    report = classification_report(y_test_smote, y_pred, output_dict=True)
    cm = confusion_matrix(y_test_smote, y_pred)

    print(f"\n{name}")
    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV weighted F1: {search.best_score_:.4f}")
    print(f"  Test weighted F1: {test_f1:.4f}")
    print(f"  Test accuracy: {accuracy:.4f}")
    print(f"  Test quadratic weighted kappa: {qwk:.4f}")
    print(f"  Test weighted ROC-AUC (OvR): {roc_auc:.4f}")
    print(f"  Confusion matrix:\n{cm}")

    return {
        'best_params': {k: str(v) for k, v in search.best_params_.items()},
        'cv_weighted_f1': search.best_score_,
        'test_weighted_f1': test_f1,
        'test_accuracy': accuracy,
        'test_quadratic_weighted_kappa': qwk,
        'test_weighted_roc_auc_ovr': roc_auc,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
    }


def save_results(results, output_dir, model_name):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"edtriage_no_nurse_{model_name}_{timestamp}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description='Replicate edtriage.ipynb RF/AdaBoost training without KTAS_RN')
    parser.add_argument('--train-path', type=str, default='../data/KTAS/edtriage_split/train.csv')
    parser.add_argument('--test-path', type=str, default='../data/KTAS/edtriage_split/test.csv')
    parser.add_argument('--output-dir', type=str, default='../results/Triage-KTAS')
    parser.add_argument('--models-dir', type=str, default='../models/edtriage_no_nurse')
    parser.add_argument('--n-iter', type=int, default=10, help='RandomizedSearchCV n_iter')
    parser.add_argument('--cv-folds', type=int, default=5)
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(script_dir, args.train_path)
    test_path = os.path.join(script_dir, args.test_path)
    output_dir = os.path.join(script_dir, args.output_dir)
    models_dir = os.path.join(script_dir, args.models_dir)
    os.makedirs(models_dir, exist_ok=True)

    np.random.seed(args.random_state)

    print("Loading data (features exclude KTAS_RN, i.e. no triage-nurse input)...")
    train_df, test_df = load_data(train_path, test_path)

    print("Building TF-IDF + SVD text features from Chief_complain...")
    text_train, text_test, tfidf_chief = build_text_pipeline(train_df, test_df)

    print("Imputing missing vitals with the median...")
    train_df, test_df = impute_missing(train_df, test_df)

    print("Removing outliers (IsolationForest, contamination=0.05) on the train set...")
    train_df = remove_outliers(train_df, random_state=args.random_state)

    print("Assembling structured + text features, SMOTE, scaling, SVD...")
    final_train_df, final_test_df, y_train_smote, y_test_smote, scaler, svd = build_features(
        train_df, test_df, text_train, text_test, random_state=args.random_state
    )
    print(f"  Final train shape: {final_train_df.shape}, test shape: {final_test_df.shape}")

    all_results = {}

    print("\n" + "=" * 60)
    print("Random Forest: RandomizedSearchCV hyperparameter tuning")
    print("=" * 60)
    rf = RandomForestClassifier()
    rs_rf = random_searchcv(rf, PARAM_GRID_RF, n_iter=args.n_iter, cv=args.cv_folds,
                             n_jobs=args.n_jobs, random_state=args.random_state)
    rs_rf.fit(final_train_df, y_train_smote)
    all_results['random_forest'] = evaluate_model('Random Forest', rs_rf, final_test_df, y_test_smote)
    with open(os.path.join(models_dir, 'random_forest.pkl'), 'wb') as f:
        pickle.dump(rs_rf.best_estimator_, f)

    print("\n" + "=" * 60)
    print("AdaBoost: RandomizedSearchCV hyperparameter tuning")
    print("=" * 60)
    ada = AdaBoostClassifier()
    rs_ada = random_searchcv(ada, PARAM_GRID_ADA, n_iter=args.n_iter, cv=args.cv_folds,
                              n_jobs=args.n_jobs, random_state=args.random_state)
    rs_ada.fit(final_train_df, y_train_smote)
    all_results['adaboost'] = evaluate_model('AdaBoost', rs_ada, final_test_df, y_test_smote)
    with open(os.path.join(models_dir, 'adaboost.pkl'), 'wb') as f:
        pickle.dump(rs_ada.best_estimator_, f)

    with open(os.path.join(models_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(models_dir, 'truncated-svd.pkl'), 'wb') as f:
        pickle.dump(svd, f)
    with open(os.path.join(models_dir, 'tf-idf-chief.pkl'), 'wb') as f:
        pickle.dump(tfidf_chief, f)

    save_results(all_results, output_dir, 'rf_adaboost')

    print("\n" + "=" * 60)
    print("Done. Models saved under:", models_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
