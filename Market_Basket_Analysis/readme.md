# Village Cinemas: F&B Analytics & Recommendation Dashboard

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red?logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-blue?logo=pandas&logoColor=white)

This project delivers an interactive dashboard for Village Cinemas, designed to perform in-depth Market Basket Analysis (MBA) on food and beverage sales. The application leverages a hybrid recommendation engine to uncover product affinities, identify promotional opportunities, and provide data-driven insights to enhance the customer experience.

### Key Features
- **Hybrid Recommendation Engine:** Combines Association Rule Mining (FP-Growth) and Collaborative Filtering (ALS) to generate robust product pairing suggestions.
- **Dynamic Contextual Filtering:** Allows users to generate highly specific recommendations by filtering the dataset on customer segments like movie genre, time of day, language, and more.
- **Performance Evaluation:** Measures the effectiveness of generated recommendations against a holdout test dataset using industry-standard metrics (Hit Rate, Precision@k, NDCG@k).
- **Scenario Explorer:** A "What If" tool to compare recommendation outcomes for two different customer segments side-by-side.

---

## 📁 Repository Structure

The project is organized to separate the data preparation pipelines from the final interactive application.

```bash
.
├── input_datasets/
│   ├── train_dataset.xlsx              # FINAL processed training data (for dashboard)
│   ├── test_dataset.xlsx               # FINAL processed holdout data (for dashboard)
│   └── inventory_transactions_clean.xlsx # FINAL master inventory map (for dashboard)
│
├── train_dataset_cleanup/
│   ├── input/                          # (Drop raw training files here)
│   ├── output/                         # (Generated files appear here)
│   └── train_dataset_cleanup.ipynb     # Cleaning Training dataset notebook 
│
├── test_dataset_cleanup/
│   ├── input/                          # (Drop raw testing files here)
│   ├── output/                         # (Generated files appear here)
│   └── test_dataset_cleanup.ipynb      # Cleaning Test dataset notebook 
│
├── models/                             # Cached models & rules (auto-created)
│
├── cinema_dashboard.py                 # The main Streamlit application script
├── market_basket_analysis.ipynb        # Exploratory analysis notebook
├── MBA Presentation.pptx               # Project presentation slides
├── requirements.txt                    # Python package dependencies
└── readme.md                           # This documentation file
```

---

## 🚀 Getting Started: Workflow & Setup

This project involves a **one-time data preparation** workflow followed by running the **interactive dashboard**. Follow these steps precisely.

### Prerequisites
- Python 3.9+
- An environment manager like `venv` or `conda` is highly recommended.

### Step 1: Setup the Environment
Clone the repository and install the required Python packages.
```bash
# Clone the repository
git clone <repository_url>
cd <repository_name>

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Data Preparation Pipeline
This two-part process cleans the raw data and prepares it for the dashboard.

#### Part A: Generate Training Data
This part processes the historical data to train the recommendation models.

1.  **Place Raw Files:** Move all raw **training** data files (e.g., `Inventory Transaction Data 2023.xlsx`, `Movie_sessions v1.0.xlsx`) into the `train_dataset_cleanup/input/` directory.
2.  **Run Notebook:** Open and execute all cells in the `train_dataset_cleanup/train_dataset_cleanup.ipynb` notebook.
3.  **➡️ Action Required:** The notebook will create several files in the `train_dataset_cleanup/output/` directory. You must **manually copy** the following two essential files to the root `input_datasets/` folder:
    - `train_dataset.xlsx`
    - `inventory_transactions_clean.xlsx` *(This serves as the master product-to-category map for the entire application).*

#### Part B: Generate Test Data
This part processes a separate, more recent set of data to evaluate the model's performance.

1.  **Place Raw Files:** Move all raw **testing** data files (e.g., `Inventory Transaction Data Feb 2025.xlsx`, `Movie_sessions_Feb2025.xlsx`) into the `test_dataset_cleanup/input/` directory.
2.  **Run Notebook:** Open and execute all cells in the `test_dataset_cleanup/test_dataset_cleanup.ipynb` notebook.
3.  **➡️ Action Required:** The notebook will generate `test_dataset.xlsx` in the `test_dataset_cleanup/output/` directory. **Manually copy** this file to the root `input_datasets/` folder.

### Step 3: Verify Final Inputs
After completing the preparation steps, your `input_datasets/` folder should contain these three files:
1.  `train_dataset.xlsx`
2.  `test_dataset.xlsx`
3.  `inventory_transactions_clean.xlsx`

### Step 4: Run the Interactive Dashboard
The Streamlit application is now ready to be launched.

1.  Navigate to the project's root directory in your terminal.
2.  Run the following command:
    ```bash
    streamlit run cinema_dashboard.py
    ```
3.  The dashboard will open in your web browser. On the first run for any new filter context, it will train the necessary models and cache them in the `models/` directory for near-instant loading on subsequent runs.

---

## 🔄 Cache Management and Refreshing Data

The dashboard heavily utilizes caching to provide a fast user experience. Models and association rules are saved to the `models/` directory based on the selected filters.

**How to Force a Refresh:**
If you update the underlying datasets in the `input_datasets/` folder or make changes to the modeling logic, the dashboard might still load the old, cached results. To force a complete regeneration of all models and rules:

1.  Open the `cinema_dashboard.py` script.
2.  Locate the `CACHE_VER` constant at the top of the file.
3.  **Increment the version string** (e.g., from `"v1.1"` to `"v1.2"`).
4.  Save the file and re-run the Streamlit app.

This will invalidate all existing caches and ensure the dashboard processes your new data from scratch.

---

## ⚙️ Core Technologies
- **Backend:** Python
- **Dashboard:** Streamlit
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:**
    - **Association Rules:** `mlxtend` (FP-Growth)
    - **Collaborative Filtering:** `implicit` (Alternating Least Squares)
- **Data Visualization:** Altair

---

## 👨‍💻 Project Context and Authors

This dashboard was developed as part of the **"Village Cinema F&B Cost and Market Basket Analysis"** project for the COSC2667/2777 Data Science and AI Postgraduate Projects course at RMIT University.

- **Team:**
    - Alen George
    - Pranjal Jhala
    - Ankith Thomas
    - Aanchal
    - Abdulrahman Asiri
    - Vinit Patnaik

- **Academic Supervisor:** Dr. Yameng Peng
- **Industry Partner:** Village Roadshow Group Services Pty Ltd