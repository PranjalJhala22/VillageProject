# Village Cinemas F&B Revenue and Sales Forecasting

This project forecasts hourly Food & Beverage (F&B) revenue and item-class sales for Village Cinemas. It delivers interactive dashboards, predictive analytics, and stocking recommendations using machine learning models built in Python.

---

## 📁 Repository Structure

```
Revenue and Sales Forecast/
├── item class breakdown/               # Scripts and data to generate product-level breakdowns
├── Model Training/                     # Jupyter notebooks and artifacts for model development
├── Prediction Script/                  # Streamlit dashboard, model loading, and inference
├── Validation scripts/                 # Final model validation and performance scripts
```

---

## ⚙️ Workflow Overview

This project involves three major stages: **Model Training**, **Prediction**, and **Validation**. Each step requires specific input data and generates relevant outputs used in the next stage.

### 1. 🏗️ Model Training

**Folder**: `Model Training/`

This phase is responsible for preparing the forecasting models and all necessary components to enable later predictions and evaluations.

#### 🔹 Inputs:
- `Inventory Transaction Data 2023 v0.1.xlsx`: Raw item-level F&B sales data for 2023
- `Inventory Transaction Data 2024 v0.1.xlsx`: Raw item-level F&B sales data for 2024
- `Movie_sessions.xlsx`: Session metadata including start times, movie info, and total admits

#### 🔹 Process:
- Preprocesses and merges inventory and session metadata
- Performs feature engineering (e.g., temporal bins, categorical encoding)
- Trains two core forecasting models:
  - **Hourly F&B Revenue Forecasting Model**
  - **Item-Class Sales Forecasting Model**
- Uses **CatBoost Regressor**, fine-tuned with RandomizedSearchCV
- Generates transformed datasets and stores final artifacts for downstream use

#### 🔹 Outputs:
- `best_catboost_model.pkl`: Trained model for revenue prediction
- `best_catboost_model_run3.pkl`: Trained model for item-class sales forecasting
- `feature_cols_run3.pkl`, `target_cols_run3.pkl`, `feature_list.pkl`: Feature and target definitions
- Intermediate transformed datasets for review/debugging

> ✅ **Important**: These `.pkl` files and related feature files must be copied to the `Prediction Script/` and `Validation scripts/` folders to enable forecasting and evaluation.

---

### 2. 📊 Prediction

**Folder**: `Prediction Script/`

This phase delivers the core functionality of the project: **forecasting future F&B revenue and sales**, and **presenting results in an interactive dashboard**.

#### 🔹 Inputs:
- Model Artifacts from `Model Training/`:
  - `best_catboost_model.pkl`
  - `best_catboost_model_run3.pkl`
  - `feature_cols_run3.pkl`
  - `target_cols_run3.pkl`
- `Movie_sessions.xlsx`: Updated session schedule with known or estimated **Total Admits**
- `Vista_Item_Percentage_Breakdown.xlsx`: Used for item-level stocking recommendations

#### 🔹 How to Run:After running the prediction ipynb file ->
```bash
cd "Prediction Script/"
streamlit run Dashboard.py
```

#### 🔹 Functionality:
- Revenue and item-class sales forecasts
- Visualizations and analytics:
  - Time-based and item-based breakdowns
  - Filters for date, hour, genre, and language
  - Stocking recommendations based on predicted item-class volumes

---

### 3. ✅ Validation

**Folder**: `Validation scripts/`

This step is for evaluating model performance by comparing forecasts with actual sales data.

#### 🔹 Inputs:
- `Movie_sessions.xlsx` (historical session data)
- `Inventory Transaction Data 2023 v0.1.xlsx`
- All model and feature files from training

#### 🔹 Process:
- Applies trained models to past data
- Simulates predictions and compares with actual sales
- Visualizes performance and error distribution

#### 🔹 Outputs:
- Metrics:
  - MAE, RMSE, R² Score
  - ±10% Accuracy bands
- Plots:
  - Actual vs. Predicted sales
  - Error distribution and trend lines
  - Item-level performance comparisons

---

## 🧠 Models Used

- **CatBoost Regressor** for both revenue and item-class forecasting
- Trained on session metadata and transactional patterns
- Tuned using `RandomizedSearchCV`

---

## 📦 Installation

### Requirements

Install dependencies using:
```bash
pip install -r requirements.txt
```

---

## 👨‍💻 Authors

Team: Pranjal Jhala, Alen George, Ankith Thomas, Aanchal, Abdulrahman Asiri, Vinit Patnaik  
Academic Supervisor: Dr. Yameng Peng  
Industry Partner: Village Roadshow Group Services Pty Ltd