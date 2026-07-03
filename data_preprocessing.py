"""
data_preprocessing.py
----------------------
Handles loading and professional-grade cleaning of the raw house price
dataset: missing values, duplicates, data types, outliers, encoding,
and scaling.

Developed by: Sapna Jabeen
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Candidate names the pipeline will search for automatically.
TARGET_CANDIDATES = ["Price", "SalePrice", "HousePrice", "price", "saleprice", "houseprice"]


class DataPreprocessor:
    """
    Encapsulates the full cleaning pipeline for the house price dataset.

    Attributes:
        df (pd.DataFrame): Working copy of the dataset.
        target_column (str): Name of the detected/assigned target column.
        label_encoders (dict): Fitted LabelEncoder objects per categorical column.
        scaler (StandardScaler): Fitted scaler for numerical features.
        summary (dict): Human-readable log of every cleaning step performed.
    """

    def __init__(self, csv_path: str, target_column: Optional[str] = None) -> None:
        """
        Args:
            csv_path: Path to the raw CSV file.
            target_column: Optional explicit target column name. If None,
                the class attempts automatic detection.
        """
        self.csv_path = Path(csv_path)
        self.df: pd.DataFrame = self._load_data()
        self.target_column: str = target_column or self._detect_target_column()
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self.summary: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_data(self) -> pd.DataFrame:
        """Loads the CSV file from disk and raises a clear error if absent."""
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at '{self.csv_path}'. "
                f"Place a CSV file inside the 'data/' folder, or run "
                f"'data/generate_sample_data.py' to create a sample dataset."
            )
        df = pd.read_csv(self.csv_path)
        logger.info("Loaded dataset with shape %s from %s", df.shape, self.csv_path)
        return df

    def _detect_target_column(self) -> str:
        """Automatically detects the target column from known candidate names."""
        for candidate in TARGET_CANDIDATES:
            if candidate in self.df.columns:
                logger.info("Auto-detected target column: '%s'", candidate)
                return candidate
        raise ValueError(
            "Could not auto-detect a target column. Expected one of "
            f"{TARGET_CANDIDATES}. Please pass `target_column` explicitly."
        )

    # ------------------------------------------------------------------
    # Cleaning steps
    # ------------------------------------------------------------------
    def remove_duplicates(self) -> "DataPreprocessor":
        """Removes exact duplicate rows."""
        before = len(self.df)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        removed = before - len(self.df)
        self.summary["duplicates_removed"] = f"{removed} duplicate row(s) removed"
        logger.info(self.summary["duplicates_removed"])
        return self

    def fix_data_types(self) -> "DataPreprocessor":
        """Attempts to coerce object columns that are numeric-like into numeric dtype."""
        fixed_cols = []
        for col in self.df.select_dtypes(include="object").columns:
            coerced = pd.to_numeric(self.df[col], errors="coerce")
            if coerced.notna().sum() / max(len(self.df), 1) > 0.9:
                self.df[col] = coerced
                fixed_cols.append(col)
        self.summary["dtype_fixes"] = f"Coerced {len(fixed_cols)} column(s) to numeric: {fixed_cols}"
        logger.info(self.summary["dtype_fixes"])
        return self

    def get_column_types(self) -> tuple[list[str], list[str]]:
        """Returns (numerical_columns, categorical_columns), excluding the target."""
        numeric_cols = [
            c for c in self.df.select_dtypes(include=[np.number]).columns
            if c != self.target_column
        ]
        categorical_cols = [
            c for c in self.df.select_dtypes(include=["object", "category"]).columns
            if c != self.target_column
        ]
        return numeric_cols, categorical_cols

    def handle_missing_values(self) -> "DataPreprocessor":
        """Imputes missing numerical values with the median and categorical with the mode."""
        numeric_cols, categorical_cols = self.get_column_types()
        missing_before = int(self.df.isna().sum().sum())

        for col in numeric_cols:
            if self.df[col].isna().any():
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)

        for col in categorical_cols:
            if self.df[col].isna().any():
                mode_val = self.df[col].mode(dropna=True)
                fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
                self.df[col] = self.df[col].fillna(fill_val)

        # Drop rows where the target itself is missing (cannot be imputed reliably).
        self.df = self.df.dropna(subset=[self.target_column]).reset_index(drop=True)

        missing_after = int(self.df.isna().sum().sum())
        self.summary["missing_values"] = (
            f"Missing values reduced from {missing_before} to {missing_after} "
            f"(numeric -> median, categorical -> mode)"
        )
        logger.info(self.summary["missing_values"])
        return self

    def handle_outliers_iqr(self, factor: float = 1.5) -> "DataPreprocessor":
        """Caps numerical outliers (including the target) using the IQR method."""
        numeric_cols, _ = self.get_column_types()
        capped_counts = {}

        for col in numeric_cols + [self.target_column]:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                logger.info("Skipping IQR capping for '%s' because IQR is zero", col)
                continue

            lower = q1 - factor * iqr
            upper = q3 + factor * iqr
            outlier_mask = (self.df[col] < lower) | (self.df[col] > upper)
            n_outliers = int(outlier_mask.sum())
            self.df.loc[outlier_mask, col] = self.df.loc[outlier_mask, col].clip(lower=lower, upper=upper)
            if n_outliers:
                capped_counts[col] = n_outliers

        self.summary["outliers_capped"] = f"Outliers capped (IQR method): {capped_counts}"
        logger.info(self.summary["outliers_capped"])
        return self

    def encode_categorical(self) -> "DataPreprocessor":
        """Label-encodes categorical columns and stores the fitted encoders for reuse."""
        _, categorical_cols = self.get_column_types()
        for col in categorical_cols:
            encoder = LabelEncoder()
            self.df[col] = encoder.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = encoder

        self.summary["encoding"] = f"Label-encoded {len(categorical_cols)} categorical column(s): {categorical_cols}"
        logger.info(self.summary["encoding"])
        return self

    def scale_numerical(self) -> "DataPreprocessor":
        """Standard-scales numerical feature columns (target excluded)."""
        numeric_cols, _ = self.get_column_types()
        self.scaler = StandardScaler()
        self.df[numeric_cols] = self.scaler.fit_transform(self.df[numeric_cols])
        self.summary["scaling"] = f"StandardScaler applied to {len(numeric_cols)} numerical column(s)"
        logger.info(self.summary["scaling"])
        return self

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def run_pipeline(self, scale: bool = True) -> pd.DataFrame:
        """Runs the full preprocessing pipeline in the correct order and returns the clean DataFrame."""
        (
            self.remove_duplicates()
            .fix_data_types()
            .handle_missing_values()
            .handle_outliers_iqr()
            .encode_categorical()
        )
        if scale:
            self.scale_numerical()
        logger.info("Preprocessing pipeline complete. Final shape: %s", self.df.shape)
        return self.df

    def print_summary(self) -> None:
        """Prints a professional preprocessing summary report."""
        print("\n" + "=" * 60)
        print("DATA PREPROCESSING SUMMARY")
        print("=" * 60)
        for step, message in self.summary.items():
            print(f"[{step}] {message}")
        print(f"[final_shape] {self.df.shape}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    dp = DataPreprocessor(csv_path="data/house_prices.csv")
    dp.run_pipeline()
    dp.print_summary()
