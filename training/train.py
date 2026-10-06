from pathlib import Path
import argparse
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42


# ============================================================
# Variables du scénario final S3
# ============================================================

FEATURES = [
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]


NUMERIC_FEATURES = [
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]


CATEGORICAL_FEATURES = [
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]


# ============================================================
# Construction du pipeline
# ============================================================

def build_pipeline():
    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="missing",
            ),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
            ),
        ),
    ])

    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            NUMERIC_FEATURES,
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES,
        ),
    ])

    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=RANDOM_STATE,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


# ============================================================
# Entraînement
# ============================================================

def train(data_path: Path, output_dir: Path):
    print(f"Chargement des données : {data_path}")

    df = pd.read_csv(
        data_path,
        sep=";",
    )

    X = df[FEATURES].copy()

    y = df["y"].map({
        "no": 0,
        "yes": 1,
    })

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    pipeline = build_pipeline()

    print("Entraînement du Gradient Boosting...")

    pipeline.fit(
        X_train,
        y_train,
    )

    # ========================================================
    # Évaluation de contrôle
    # ========================================================

    y_proba = pipeline.predict_proba(
        X_test
    )[:, 1]

    y_pred = (
        y_proba >= 0.5
    ).astype(int)

    metrics = {
        "precision_0.5": float(
            precision_score(y_test, y_pred)
        ),
        "recall_0.5": float(
            recall_score(y_test, y_pred)
        ),
        "f1_0.5": float(
            f1_score(y_test, y_pred)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, y_proba)
        ),
    }

    print("\nMétriques de contrôle")
    print("---------------------")

    for name, value in metrics.items():
        print(
            f"{name}: {value:.4f}"
        )

    # ========================================================
    # Sauvegarde
    # ========================================================

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        output_dir
        / "pipeline.joblib"
    )

    joblib.dump(
        pipeline,
        model_path,
    )

    metadata = {
        "model": "GradientBoostingClassifier",
        "scenario": "S3",
        "random_state": RANDOM_STATE,
        "features": FEATURES,
        "metrics_test": metrics,
    }

    metadata_path = (
        output_dir
        / "model_metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nModèle sauvegardé : {model_path}"
    )

    print(
        f"Métadonnées sauvegardées : {metadata_path}"
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Entraînement du modèle "
            "Bank Marketing"
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Chemin vers bank-additional-full.csv",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models"),
        help="Dossier de sortie du modèle",
    )

    args = parser.parse_args()

    train(
        data_path=args.data,
        output_dir=args.output,
    )