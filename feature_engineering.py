"""
feature_engineering.py
------------------------
Feature creation, skewness correction, feature selection, and
train/test splitting for the House Price Prediction System.

Developed by: Sapna Jabeen
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Encapsulates feature-engineering steps applied after preprocessing:
    dropping irrelevant columns, fixing skewness, selecting the most
    predictive features, and producing a train/test split.
    """

    def __init__(self, df: pd.DataFrame, target_column: str) -> None:
        """
        Args:
            df: Cleaned DataFrame (output of DataPreprocessor).
            target_column: Name of the target/label column.
        """
        self.df = df.copy()
        self.target_column = target_column
        self.selected_features: list[str] = []

    def drop_irrelevant_columns(self, columns: Optional[list[str]] = None) -> "FeatureEngineer":
        """Drops explicitly irrelevant columns such as ID fields, if present."""
        default_drop = ["Id", "ID", "id", "index"]
        cols_to_drop = [c for c in (columns or default_drop) if c in self.df.columns]
        if cols_to_drop:
            self.df = self.df.drop(columns=cols_to_drop)
            logger.info("Dropped irrelevant columns: %s", cols_to_drop)
        return self

    def handle_skewness(self, threshold: float = 0.75) -> "FeatureEngineer":
        """Applies a log1p transform to numerical features with high skewness."""
        numeric_cols = [
            c for c in self.df.select_dtypes(include=[np.number]).columns
            if c != self.target_column
        ]
        transformed = []
        for col in numeric_cols:
            col_skew = skew(self.df[col].dropna())
            if abs(col_skew) > threshold and (self.df[col] >= 0).all():
                self.df[col] = np.log1p(self.df[col])
                transformed.append(col)
        logger.info("Applied log1p transform to skewed features: %s", transformed)
        return self

    def select_features(self, k: int = 10) -> "FeatureEngineer":
        """Selects the top-k most predictive features using an F-regression test."""
        feature_cols = [c for c in self.df.columns if c != self.target_column]
        k = min(k, len(feature_cols))

        X = self.df[feature_cols]
        y = self.df[self.target_column]

        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X, y)
        mask = selector.get_support()
        self.selected_features = list(np.array(feature_cols)[mask])
        logger.info("Selected top %d features: %s", k, self.selected_features)
        return self

    def train_test_split(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Splits the dataset into train and test sets using the selected features."""
        features = self.selected_features if self.selected_features else [
            c for c in self.df.columns if c != self.target_column
        ]
        X = self.df[features]
        y = self.df[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        logger.info(
            "Train/test split complete. Train: %s | Test: %s (test_size=%.2f, random_state=%d)",
            X_train.shape, X_test.shape, test_size, random_state,
        )
        return X_train, X_test, y_train, y_test

    def run_pipeline(
        self, k_features: int = 10, test_size: float = 0.2, random_state: int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Runs the complete feature engineering pipeline end to end."""
        self.drop_irrelevant_columns().handle_skewness().select_features(k=k_features)
        return self.train_test_split(test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor

    dp = DataPreprocessor(csv_path="data/house_prices.csv")
    clean_df = dp.run_pipeline()

    fe = FeatureEngineer(clean_df, target_column=dp.target_column)
    X_train, X_test, y_train, y_test = fe.run_pipeline()
    print(f"Selected features: {fe.selected_features}")
    print(f"X_train shape: {X_train.shape} | X_test shape: {X_test.shape}")
