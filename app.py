"""

app.py
-------
Professional Streamlit web application for the House Price Prediction
System. Provides an interactive UI for making predictions, exploring
the dataset, viewing visualizations, and comparing model performance.

Run with:
    streamlit run app.py

Developed by: Sapna Jabeen
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from predict import HousePricePredictor  # noqa: E402
from data_preprocessing import DataPreprocessor  # noqa: E402
from feature_engineering import FeatureEngineer  # noqa: E402

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="House Price Prediction System",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS_DIR = Path("models")
DATA_PATH = Path("data/house_prices.csv")

# ----------------------------------------------------------------------
# Custom CSS — beautiful, professional theme
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main { background-color: #f7f9fc; }

    .hero {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 55%, #6a8caf 100%);
        padding: 2.4rem 2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.6rem;
        box-shadow: 0 8px 24px rgba(30, 60, 114, 0.25);
    }
    .hero h1 { font-size: 2.3rem; margin-bottom: 0.3rem; font-weight: 800; }
    .hero p { font-size: 1.05rem; opacity: 0.92; margin: 0; }

    .credit-badge {
        display: inline-block;
        background: rgba(255,255,255,0.18);
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 0.85rem;
        margin-top: 0.8rem;
        border: 1px solid rgba(255,255,255,0.35);
    }

    .price-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.8rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 20px rgba(17, 153, 142, 0.3);
        margin-top: 1rem;
    }
    .price-card h2 { font-size: 2.4rem; margin: 0.2rem 0 0 0; font-weight: 800; }
    .price-card p { margin: 0; opacity: 0.9; font-size: 0.95rem; letter-spacing: 0.5px; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #16213e 100%);
    }
    section[data-testid="stSidebar"] * { color: #f0f2f6 !important; }

    .footer-note {
        text-align: center;
        color: #6c757d;
        font-size: 0.85rem;
        padding: 1.2rem 0 0.4rem 0;
        border-top: 1px solid #e0e0e0;
        margin-top: 2rem;
    }

    div[data-testid="stMetric"] {
        background: white;
        padding: 0.9rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_predictor() -> HousePricePredictor | None:
    """Loads the persisted predictor, returning None if artifacts are missing."""
    try:
        return HousePricePredictor()
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame | None:
    """Loads the raw dataset for preview/EDA purposes."""
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None


def get_test_predictions(predictor: HousePricePredictor, model_name: str):
    """Recomputes test-set predictions vs actuals for a given model (for plotting)."""
    dp = DataPreprocessor(csv_path=str(DATA_PATH), target_column=predictor.target_column)
    clean_df = dp.run_pipeline()
    fe = FeatureEngineer(clean_df, target_column=predictor.target_column)
    fe.selected_features = predictor.selected_features
    X_train, X_test, y_train, y_test = fe.train_test_split()

    file_map = {"Linear Regression": "linear_regression.pkl", "Random Forest": "random_forest.pkl"}
    model = joblib.load(MODELS_DIR / file_map[model_name])
    y_pred = model.predict(X_test)
    return y_test, y_pred, model


# ----------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏠 Navigation")
    page = st.radio(
        "Go to",
        ["🔮 Predict Price", "📊 Dataset Preview", "📈 Visualizations", "🧪 Model Evaluation"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "This application predicts house prices using trained "
        "**Linear Regression** and **Random Forest** models, built on "
        "a full data science pipeline: cleaning → EDA → feature "
        "engineering → training → evaluation."
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.85rem; opacity:0.85;'>"
        "👩‍💻 <b>Developed by</b><br>Sapna Jabeen</div>",
        unsafe_allow_html=True,
    )

predictor = load_predictor()
raw_df = load_dataset()

# ----------------------------------------------------------------------
# Hero header
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🏠 House Price Prediction System</h1>
        <p>An end-to-end Machine Learning application for estimating property prices,
        powered by Linear Regression &amp; Random Forest models.</p>
        <span class="credit-badge">👩‍💻 Developed by Sapna Jabeen</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if predictor is None:
    st.error(
        "⚠️ No trained model artifacts found. Please run `python main.py` "
        "from the project root first to train and save the models."
    )
    st.stop()

# ========================================================================
# PAGE 1 — PREDICT PRICE
# ========================================================================
if page == "🔮 Predict Price":
    st.subheader("🔮 Predict a House Price")

    col_form, col_result = st.columns([1.4, 1], gap="large")

    with col_form:
        with st.container():
            st.markdown("#### Model Selection")
            model_choice = st.selectbox(
                "Choose the model used for prediction",
                ["Best Model (Recommended)", "Linear Regression", "Random Forest"],
            )

            st.markdown("#### Property Details")

            # Automatically generate input fields based on the selected features
            input_values: dict = {}
            categorical_options = {
                col: list(enc.classes_) for col, enc in predictor.label_encoders.items()
            }

            field_cols = st.columns(2)
            defaults = {
                "SquareFeet": 1800, "Bedrooms": 3, "Bathrooms": 2.0, "YearBuilt": 2005,
                "LotSize": 6000, "GarageCars": 2, "OverallQuality": 6, "HasPool": 0,
                "DistanceToCityCenter_km": 5.0, "Stories": 2,
            }

            for i, feature in enumerate(predictor.selected_features):
                target_col = field_cols[i % 2]
                with target_col:
                    if feature in categorical_options:
                        input_values[feature] = st.selectbox(feature, categorical_options[feature])
                    elif feature == "HasPool":
                        input_values[feature] = 1 if st.checkbox("Has Pool") else 0
                    elif feature in ("Bedrooms", "GarageCars", "Stories", "OverallQuality", "YearBuilt"):
                        input_values[feature] = st.number_input(
                            feature, min_value=0, value=int(defaults.get(feature, 0)), step=1
                        )
                    else:
                        input_values[feature] = st.number_input(
                            feature, min_value=0.0, value=float(defaults.get(feature, 0.0)), step=1.0
                        )

            predict_clicked = st.button("💰 Predict Price", use_container_width=True)

    with col_result:
        if predict_clicked:
            try:
                if model_choice == "Best Model (Recommended)":
                    predictor.load_model_by_name(predictor.best_model_name)
                    used_model = predictor.best_model_name
                else:
                    predictor.load_model_by_name(model_choice)
                    used_model = model_choice

                price = predictor.predict(input_values)

                st.markdown(
                    f"""
                    <div class="price-card">
                        <p>ESTIMATED HOUSE PRICE</p>
                        <h2>${price:,.0f}</h2>
                        <p style="margin-top:0.6rem;">Model used: {used_model}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                r2 = predictor.results[used_model]["R2 Score"]
                st.metric("Model R² Score", f"{r2:.3f}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Prediction failed: {exc}")
        else:
            st.info("Fill in the property details and click **Predict Price** to see the estimate.")

# ========================================================================
# PAGE 2 — DATASET PREVIEW
# ========================================================================
elif page == "📊 Dataset Preview":
    st.subheader("📊 Dataset Preview")

    if raw_df is None:
        st.warning("Dataset not found at `data/house_prices.csv`.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{raw_df.shape[0]:,}")
        c2.metric("Columns", raw_df.shape[1])
        c3.metric("Missing Values", int(raw_df.isna().sum().sum()))
        c4.metric("Duplicate Rows", int(raw_df.duplicated().sum()))

        with st.expander("🔍 View Raw Data", expanded=True):
            st.dataframe(raw_df.head(50), use_container_width=True)

        with st.expander("📈 Summary Statistics"):
            st.dataframe(raw_df.describe().T, use_container_width=True)

        with st.expander("🗂️ Column Data Types"):
            st.dataframe(raw_df.dtypes.astype(str).rename("dtype"), use_container_width=True)

# ========================================================================
# PAGE 3 — VISUALIZATIONS
# ========================================================================
elif page == "📈 Visualizations":
    st.subheader("📈 Data Visualizations")

    if raw_df is None:
        st.warning("Dataset not found at `data/house_prices.csv`.")
    else:
        numeric_df = raw_df.select_dtypes(include=[np.number])
        target_col = predictor.target_column

        tab1, tab2, tab3 = st.tabs(["Correlation Heatmap", "Target Distribution", "Feature Relationships"])

        with tab1:
            fig, ax = plt.subplots(figsize=(9, 6))
            sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.5)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)
            st.caption(
                "Features with strong positive correlation to the target tend to be the "
                "most influential drivers of price, such as square footage and overall quality."
            )

        with tab2:
            col_a, col_b = st.columns(2)
            with col_a:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(raw_df[target_col], kde=True, ax=ax, color="#2a5298")
                ax.set_title(f"{target_col} Distribution")
                st.pyplot(fig)
            with col_b:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.boxplot(x=raw_df[target_col], ax=ax, color="#38ef7d")
                ax.set_title(f"{target_col} Boxplot")
                st.pyplot(fig)
            st.caption(
                "The distribution shows the overall spread of house prices, while the "
                "boxplot highlights potential outliers beyond the interquartile range."
            )

        with tab3:
            numeric_features = [c for c in numeric_df.columns if c != target_col]
            selected_feature = st.selectbox("Select a feature to compare against the target", numeric_features)
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=raw_df, x=selected_feature, y=target_col, ax=ax, alpha=0.6, color="#1e3c72")
            ax.set_title(f"{selected_feature} vs {target_col}")
            st.pyplot(fig)

# ========================================================================
# PAGE 4 — MODEL EVALUATION
# ========================================================================
elif page == "🧪 Model Evaluation":
    st.subheader("🧪 Model Evaluation & Comparison")

    results_df = pd.DataFrame(predictor.results).T
    st.markdown("#### Metrics Comparison")
    st.dataframe(results_df.style.highlight_max(subset=["R2 Score", "CV R2 Mean"], color="#c6f6d5"),
                 use_container_width=True)

    best = predictor.best_model_name
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best Model", best)
    m2.metric("R² Score", f"{predictor.results[best]['R2 Score']:.4f}")
    m3.metric("RMSE", f"${predictor.results[best]['RMSE']:,.0f}")
    m4.metric("MAE", f"${predictor.results[best]['MAE']:,.0f}")

    st.markdown("#### Visual Diagnostics")
    model_for_plot = st.selectbox("Select model to visualize", list(predictor.results.keys()))

    if raw_df is not None:
        y_test, y_pred, _ = get_test_predictions(predictor, model_for_plot)

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(y_test, y_pred, alpha=0.5, color="#2a5298")
            lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
            ax.plot(lims, lims, "r--", linewidth=1.5)
            ax.set_xlabel("Actual Price")
            ax.set_ylabel("Predicted Price")
            ax.set_title("Prediction vs Actual")
            st.pyplot(fig)

        with col2:
            residuals = y_test - y_pred
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.histplot(residuals, kde=True, ax=ax, color="#e07a5f")
            ax.set_title("Residual / Error Distribution")
            ax.set_xlabel("Residual (Actual - Predicted)")
            st.pyplot(fig)

        if model_for_plot == "Random Forest":
            rf_model = joblib.load(MODELS_DIR / "random_forest.pkl")
            importances = pd.Series(rf_model.feature_importances_, index=predictor.selected_features)
            importances = importances.sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(7, 5))
            importances.plot(kind="barh", ax=ax, color="#11998e")
            ax.set_title("Feature Importance (Random Forest)")
            st.pyplot(fig)

        st.markdown("#### Model Comparison Chart")
        fig, ax = plt.subplots(figsize=(7, 4))
        results_df["R2 Score"].plot(kind="bar", ax=ax, color=["#11998e", "#2a5298"])
        ax.set_ylabel("R² Score")
        ax.set_title("Model Comparison — R² Score")
        plt.xticks(rotation=0)
        st.pyplot(fig)

# ----------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="footer-note">
        🏠 House Price Prediction System &nbsp;|&nbsp; Built with Python, Scikit-Learn &amp; Streamlit
        <br>👩‍💻 Developed by <b>Sapna Jabeen</b>
    </div>
    """,
    unsafe_allow_html=True,
)
