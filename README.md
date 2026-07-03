# 🏠 House Price Prediction System

An end-to-end **Machine Learning** project that predicts residential house prices using regression techniques, complete with data cleaning, exploratory data analysis, feature engineering, model training/evaluation, and a professional interactive **Streamlit** web application.

**Developed by:** Sapna Jabeen

---

## 📌 Project Overview

This project simulates a real-world Data Science workflow used by industry ML teams:

1. **Data Cleaning** — missing values, duplicates, data types, IQR-based outlier handling
2. **Exploratory Data Analysis (EDA)** — statistical summaries and visual interpretation
3. **Feature Engineering** — skewness correction, feature selection, train/test splitting
4. **Model Training** — Linear Regression & Random Forest Regressor
5. **Model Evaluation** — MAE, MSE, RMSE, R², cross-validation, automatic best-model selection
6. **Deployment** — an interactive Streamlit application for live predictions and visualization

It is built to be modular, readable, and portfolio-ready.

---

## ✨ Features

- 🔍 Automatic target column detection (`Price`, `SalePrice`, `HousePrice`)
- 🧹 Production-quality preprocessing pipeline (`DataPreprocessor` class)
- 🛠️ Configurable feature engineering pipeline (`FeatureEngineer` class)
- 🤖 Dual-model training with automatic best-model selection based on R²
- 💾 Persisted models & preprocessing artifacts via **Joblib**
- 📊 Rich visualizations: correlation heatmaps, residual plots, feature importance, model comparison charts
- 🌐 Beautiful, professional **Streamlit** web app with:
  - Auto-generated prediction input form
  - Model selector (Linear Regression / Random Forest / Best Model)
  - Dataset preview & summary statistics
  - Interactive visualizations
  - Full model evaluation dashboard
- ✅ PEP8-compliant, type-hinted, documented, and exception-safe code

---

## 📂 Dataset

By default, this project ships with a **synthetically generated but realistic** house price dataset (`data/house_prices.csv`), created via `data/generate_sample_data.py`. It includes features such as square footage, bedrooms, bathrooms, year built, lot size, garage capacity, neighborhood, overall quality, pool presence, distance to city center, and number of stories — along with intentionally injected missing values, duplicates, and outliers so the cleaning pipeline has real work to do.

**To use your own dataset:**

1. Place your CSV file inside the `data/` folder.
2. Ensure the target column is named `Price`, `SalePrice`, or `HousePrice` (or pass `--target <column_name>` to `main.py`).
3. Run the pipeline as usual — everything else is handled automatically.

---

## 🗂️ Project Structure

```
House_Price_Prediction/
│
├── data/
│   ├── house_prices.csv          # Dataset (place your own CSV here)
│   └── generate_sample_data.py   # Synthetic dataset generator
│
├── notebooks/
│   └── EDA.ipynb                 # Exploratory Data Analysis notebook
│
├── src/
│   ├── data_preprocessing.py     # Cleaning: missing values, duplicates, outliers, encoding, scaling
│   ├── feature_engineering.py    # Feature creation, skewness fix, selection, train/test split
│   ├── train_model.py            # Model training, evaluation, artifact saving
│   └── predict.py                # Inference on new raw input data
│
├── models/                       # Saved models & pipeline artifacts (generated after training)
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   ├── best_model.pkl
│   └── pipeline_artifacts.pkl
│
├── app.py                        # Streamlit web application
├── main.py                       # Full pipeline entry point (training)
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd House_Price_Prediction

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. (Optional) Generate the sample dataset
```bash
cd data
python generate_sample_data.py
cd ..
```
Skip this step if you've placed your own CSV in `data/`.

### 2. Train the models
```bash
python main.py
```
This runs the full pipeline — preprocessing, feature engineering, training, evaluation — and saves all artifacts to `models/`.

Optional arguments:
```bash
python main.py --data data/house_prices.csv --target SalePrice --k-features 10 --test-size 0.2
```

### 3. Launch the Streamlit app
```bash
streamlit run app.py
```
Then open the local URL shown in your terminal (typically `http://localhost:8501`).

### 4. Explore the EDA notebook
```bash
jupyter notebook notebooks/EDA.ipynb
```

---

## 📈 Model Performance

*(Example results from the included sample dataset — your numbers will vary with your own data.)*

| Model              | MAE       | RMSE      | R² Score | CV R² Mean |
|---------------------|-----------|-----------|----------|------------|
| **Random Forest**   | ~23,700   | ~29,800   | ~0.943   | ~0.900     |
| Linear Regression   | ~62,900   | ~77,800   | ~0.614   | ~0.607     |

The pipeline automatically selects and persists the model with the highest R² score as `models/best_model.pkl`.

---

## 🖼️ Screenshots

> _Add screenshots of the Streamlit app here after running it locally._

- `screenshots/predict_page.png` — Prediction interface
- `screenshots/dataset_preview.png` — Dataset preview page
- `screenshots/visualizations.png` — Visualizations dashboard
- `screenshots/model_evaluation.png` — Model evaluation dashboard

---

## 🔮 Future Improvements

- Add gradient boosting models (XGBoost, LightGBM, CatBoost) to the comparison
- Hyperparameter tuning via GridSearchCV / Optuna
- SHAP-based model explainability
- Geospatial features (latitude/longitude, map visualizations)
- CI/CD pipeline with automated testing (pytest)
- Dockerize the application for one-command deployment
- REST API endpoint (FastAPI) alongside the Streamlit UI

---

## 👩‍💻 Developer

**Sapna Jabeen**
Data Analytics / Machine Learning Intern — Week 1 Project

---

## 📄 License

This project is provided for educational and portfolio purposes.
