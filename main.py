"""
main.py
--------
Single entry point that runs the complete House Price Prediction
pipeline end to end: preprocessing -> feature engineering -> training
-> evaluation -> artifact saving.

Usage:
    python main.py
    python main.py --data data/house_prices.csv --k-features 10

Developed by: Sapna Jabeen
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_preprocessing import DataPreprocessor  # noqa: E402
from feature_engineering import FeatureEngineer  # noqa: E402
from train_model import ModelTrainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the pipeline run."""
    parser = argparse.ArgumentParser(description="House Price Prediction — full training pipeline")
    parser.add_argument("--data", type=str, default="data/house_prices.csv", help="Path to the raw CSV dataset")
    parser.add_argument("--target", type=str, default=None, help="Target column name (auto-detected if omitted)")
    parser.add_argument("--k-features", type=int, default=10, help="Number of top features to select")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split proportion")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory to save trained artifacts")
    return parser.parse_args()


def main() -> None:
    """Runs the full end-to-end training pipeline and prints a final report."""
    args = parse_args()

    print("=" * 70)
    print(" HOUSE PRICE PREDICTION SYSTEM — TRAINING PIPELINE")
    print(" Developed by: Sapna Jabeen")
    print("=" * 70)

    # Step 1: Preprocessing
    print("\n[1/4] Cleaning and preprocessing data...")
    preprocessor = DataPreprocessor(csv_path=args.data, target_column=args.target)
    clean_df = preprocessor.run_pipeline()
    preprocessor.print_summary()

    # Step 2: Feature Engineering
    print("[2/4] Engineering features and splitting data...")
    engineer = FeatureEngineer(clean_df, target_column=preprocessor.target_column)
    X_train, X_test, y_train, y_test = engineer.run_pipeline(
        k_features=args.k_features, test_size=args.test_size, random_state=args.random_state
    )
    print(f"      Selected features: {engineer.selected_features}")
    print(f"      Train shape: {X_train.shape} | Test shape: {X_test.shape}")

    # Step 3: Training + Evaluation
    print("\n[3/4] Training models (Linear Regression & Random Forest)...")
    trainer = ModelTrainer()
    comparison = trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
    print("\nMODEL COMPARISON TABLE")
    print(comparison.round(4))
    print(f"\n Best model: {trainer.best_model_name}")

    # Step 4: Save artifacts
    print("\n[4/4] Saving trained models and pipeline artifacts...")
    trainer.save_artifacts(
        target_column=preprocessor.target_column,
        selected_features=engineer.selected_features,
        label_encoders=preprocessor.label_encoders,
        scaler=preprocessor.scaler,
        models_dir=args.models_dir,
    )

    print("\n" + "=" * 70)
    print(" PIPELINE COMPLETE — run 'streamlit run app.py' to launch the app")
    print("=" * 70)


if __name__ == "__main__":
    main()
