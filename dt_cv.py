#!/usr/bin/env python3
"""
Decision Tree (DT) with 10-fold Stratified CV + SMOTE-Tomek (k=5), reporting Accuracy, Precision (macro), Recall (macro), F1 (macro).

Usage:
  python dt_cv.py --data path/to/data.csv --target TARGET_COLUMN

Notes:
- Categorical columns are one-hot encoded automatically.
- Missing values are imputed with SimpleImputer (most_frequent for categoricals, median for numerics).
- SMOTE-Tomek is applied *inside* each CV fold to avoid leakage.
- Metrics are macro-averaged to account for class imbalance.
"""

import argparse
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.model_selection import StratifiedKFold, cross_validate

from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

from sklearn.tree import DecisionTreeClassifier

def build_model():
    # Instantiate the classifier with the hyperparameters from the table
    clf = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, random_state=42)
    return clf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--target', type=str, required=True, help='Target column name')
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{{args.target}}' not found in columns: {{list(df.columns)}}")

    y = df[args.target]
    X = df.drop(columns=[args.target])

    # Identify column types
    cat_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()
    num_cols = X.select_dtypes(include=['number']).columns.tolist()

    # Preprocess: impute + one-hot for categoricals; impute median for numerics
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), num_cols),
            ('cat', SkPipeline(steps=[
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
            ]), cat_cols),
        ],
        remainder='drop'
    )

    # SMOTE-Tomek settings
    smt = SMOTETomek(
        sampling_strategy='auto',
        smote=SMOTE(k_neighbors=5, random_state=42),
        tomek=TomekLinks()
    )

    # Full pipeline: preprocessing -> SMOTE-Tomek -> classifier
    clf = build_model()
    pipe = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('balance', smt),
        ('model', clf)
    ])

    # Stratified 10-fold CV
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, average='macro', zero_division=0),
        'recall': make_scorer(recall_score, average='macro', zero_division=0),
        'f1': make_scorer(f1_score, average='macro', zero_division=0),
    }

    results = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)

    def mean_std(arr):
        import numpy as _np
        return float(_np.mean(arr)), float(_np.std(arr))

    metrics_summary = {{
        'accuracy_mean': mean_std(results['test_accuracy'])[0],
        'accuracy_std': mean_std(results['test_accuracy'])[1],
        'precision_mean': mean_std(results['test_precision'])[0],
        'precision_std': mean_std(results['test_precision'])[1],
        'recall_mean': mean_std(results['test_recall'])[0],
        'recall_std': mean_std(results['test_recall'])[1],
        'f1_mean': mean_std(results['test_f1'])[0],
        'f1_std': mean_std(results['test_f1'])[1],
    }}

    print("===== Decision Tree (DT) (10-fold CV) =====")
    for k, v in metrics_summary.items():
        print(f"{{k}}: {{v:.4f}}")

    # Also print per-fold metrics (optional)
    print("\nPer-fold metrics:")
    for i in range(len(results['test_accuracy'])):
        print(f"Fold {{i+1}} -> acc={{results['test_accuracy'][i]:.4f}}, "
              f"prec={{results['test_precision'][i]:.4f}}, "
              f"rec={{results['test_recall'][i]:.4f}}, "
              f"f1={{results['test_f1'][i]:.4f}}")

if __name__ == '__main__':
    main()
