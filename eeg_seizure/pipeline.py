"""Classification pipelines and cross-validation for EEG feature matrices."""
from __future__ import annotations
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def make_classifiers(k_features=12):
    """Return {name: sklearn Pipeline} with scaling + feature selection."""
    base = lambda clf: Pipeline([
        ("scale", StandardScaler()),
        ("select", SelectKBest(f_classif, k=k_features)),
        ("clf", clf),
    ])
    return {
        "KNN": base(KNeighborsClassifier(n_neighbors=5)),
        "LDA": base(LinearDiscriminantAnalysis()),
        "SVM (RBF)": base(SVC(kernel="rbf", C=10, gamma="scale")),
        "Decision Tree": base(DecisionTreeClassifier(max_depth=6, random_state=0)),
        "Random Forest": base(RandomForestClassifier(n_estimators=200, random_state=0)),
    }


def evaluate(X, y, k_features=12, n_splits=10, seed=0):
    """Stratified k-fold CV for every classifier.

    Returns {name: {"accuracy", "y_pred"}}.
    """
    k_features = min(k_features, X.shape[1])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = {}
    for name, pipe in make_classifiers(k_features).items():
        y_pred = cross_val_predict(pipe, X, y, cv=cv)
        results[name] = {"accuracy": accuracy_score(y, y_pred), "y_pred": y_pred}
    return results
