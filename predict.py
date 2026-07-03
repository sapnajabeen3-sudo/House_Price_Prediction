"""
predict.py
-----------
Loads the saved model + pipeline artifacts and provides a simple
interface for predicting house prices on new, raw input data.

Developed by: Sapna Jabeen
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")


class HousePricePredictor:
    """Loads persisted artifacts and serves predictions for new raw input rows."""

    def __init__(self, models_dir: str | Path = MODELS_DIR, use_best: bool = True) -> None:
        """
        Args:
            models_dir: Directory containing the saved .pkl artifacts.
            use_best: If True, loads best_model.pkl; otherwise the caller
                can load a specific model via `load_model_by_name`.
        """
        models_dir = Path(models_dir)
        artifacts_path = models_dir / "pipeline_artifacts.pkl"
        if not artifacts_path.exists():
            raise FileNotFoundError(
                f"Artifacts not found at '{artifacts_path}'. Run src/train_model.py first."
            )

        artifacts = joblib.load(artifacts_path)
        self.target_column: str = artifacts["target_column"]
        self.selected_features: list[str] = artifacts["selected_features"]
        self.label_encoders: dict[str, Any] = artifacts["label_encoders"]
        self.scaler = artifacts["scaler"]
        self.best_model_name: str = artifacts["best_model_name"]
        self.results: dict = artifacts["results"]

        self.model = (
            joblib.load(models_dir / "best_model.pkl")
            if use_best
            else None
        )
        self.models_dir = models_dir

    def load_model_by_name(self, model_name: str) -> None:
        """Swaps the active model, e.g. 'Linear Regression' or 'Random Forest'."""
        file_map = {
            "Linear Regression": "linear_regression.pkl",
            "Random Forest": "random_forest.pkl",
        }
        if model_name not in file_map:
            raise ValueError(f"Unknown model_name '{model_name}'. Choose from {list(file_map)}.")
        self.model = joblib.load(self.models_dir / file_map[model_name])
        logger.info("Loaded model: %s", model_name)

    def _prepare_input(self, raw_input: dict) -> pd.DataFrame:
        """Encodes and scales a single raw input dict into model-ready features.

        Note: the scaler was fit on ALL numeric/encoded columns present at
        training time, but `selected_features` (chosen by SelectKBest) may be
        a *subset* of those columns. To call `scaler.transform()` safely we
        must supply a row with exactly the columns/order the scaler expects
        (`scaler.feature_names_in_`). Any column the caller didn't provide
        (i.e. one that wasn't selected as a model feature) is filled with
        that column's training-set mean, so after scaling it becomes ~0 —
        a neutral value that doesn't distort the prediction.
        """
        row_dict: dict = dict(raw_input)

        # Apply the same label encoders used during training.
        for col, encoder in self.label_encoders.items():
            if col in row_dict:
                value = str(row_dict[col])
                if value in encoder.classes_:
                    row_dict[col] = int(encoder.transform([value])[0])
                else:
                    # Unseen category -> fall back to the most frequent class.
                    logger.warning("Unseen category '%s' for column '%s'; using fallback.", value, col)
                    row_dict[col] = int(encoder.transform([encoder.classes_[0]])[0])

        if self.scaler is not None and hasattr(self.scaler, "feature_names_in_"):
            scaler_columns = list(self.scaler.feature_names_in_)
            training_means = dict(zip(scaler_columns, self.scaler.mean_))

            # Build a row containing every column the scaler was fit on,
            # filling gaps with the training mean.
            row = pd.DataFrame(
                [{col: row_dict.get(col, float(training_means.get(col, 0.0))) for col in scaler_columns}]
            )
            row[scaler_columns] = self.scaler.transform(row[scaler_columns])

            # Preserve categorical or other selected features that were not scaled.
            for col in self.selected_features:
                if col not in row.columns:
                    row[col] = row_dict.get(col, 0)
        else:
            row = pd.DataFrame([{col: row_dict.get(col, 0) for col in self.selected_features}])

        return row[self.selected_features]

    def predict(self, raw_input: dict) -> float:
        """Predicts a single house price from a dict of raw feature values."""
        if self.model is None:
            raise RuntimeError("No model loaded. Initialize with use_best=True or call load_model_by_name().")
        X = self._prepare_input(raw_input)
        prediction = self.model.predict(X)[0]
        return float(prediction)


if __name__ == "__main__":
    predictor = HousePricePredictor()
    sample_input = {
        "SquareFeet": 2200,
        "Bedrooms": 3,
        "Bathrooms": 2.5,
        "YearBuilt": 2005,
        "LotSize": 6500,
        "GarageCars": 2,
        "Neighborhood": "Suburb",
        "OverallQuality": 7,
        "HasPool": 0,
        "DistanceToCityCenter_km": 5.2,
        "Stories": 2,
    }
    price = predictor.predict(sample_input)
    print(f"Predicted price using {predictor.best_model_name}: ${price:,.2f}")
