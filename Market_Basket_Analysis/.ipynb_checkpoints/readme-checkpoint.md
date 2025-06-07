
# Village Cinemas: F&B Market Basket Analysis & Recommendation App

This project provides a comprehensive Market Basket Analysis (MBA) and an interactive Streamlit dashboard for Village Cinemas. The application helps identify F&B product associations, explore data-driven combo opportunities, and understand customer purchasing patterns through a dynamic, filterable interface.

---

## 📁 Repository Structure

```bash
Village_Cinemas_MBA/
│
├── Input Datasets/
│   ├── train_dataset.xlsx              # FINAL processed training data
│   ├── test_dataset.xlsx               # FINAL processed holdout data
│   └── inventory_transactions_clean.xlsx # FINAL cleaned inventory map
│
├── Train Dataset Cleanup/
│   ├── input/
│   │   └── (Place all raw training Excel files here)
│   ├── output/
│   │   ├── train_dataset.xlsx
│   │   ├── inventory_transactions_clean.xlsx
│   │   └── (Other intermediate files...)
│   └── Village Roadshows - Training Dataset Cleanup.ipynb
│
├── Test Dataset Cleanup/
│   ├── input/
│   │   └── (Place all raw test period Excel files here)
│   ├── output/
│   │   ├── test_dataset.xlsx
│   │   └── (Other intermediate files...)
│   └── Village Roadshows - Test Dataset Cleanup.ipynb
│
├── models/                             # Cached models & rules (auto-created by dashboard)
│
├── cinema_dashboard.py                 # The Streamlit application script
├── Market Basket Analysis.ipynb        # (Optional) Exploratory MBA notebook
├── MBA Presentation.pptx               # Project presentation slides
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

---

## ⚙️ Workflow & Setup

This project is divided into two main parts: a **one-time data preparation pipeline** to create the analysis-ready datasets, and the **interactive Streamlit dashboard** that consumes these datasets.

### 1. Prerequisites
- Python 3.9+
- Pip (Python package installer)
- Jupyter Notebook or JupyterLab

### 2. Setup
   a. Clone this repository to your local machine.
   b. It is highly recommended to create a Python virtual environment:
      ```bash
      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate
      ```
   c. Install the required Python packages from the project root:
      ```bash
      pip install -r requirements.txt
      ```

### 3. Data Preparation Pipeline

This is a two-stage process to create the final `train_dataset.xlsx` and `test_dataset.xlsx`.

#### Stage 1: Generate the Training Dataset
   a. **Place Raw Data:** Add all raw training data files (e.g., `Inventory Transaction Data 2023`, `Inventory Transaction Data 2024`, `Movie_sessions.xlsx`, etc.) into the `Train Dataset Cleanup/input/` folder.
   b. **Run Notebook:** Open and run all cells in the `Train Dataset Cleanup/Village Roadshows - Training Dataset Cleanup.ipynb` notebook.
   c. **Action Required:** This notebook generates several output files in `Train Dataset Cleanup/output/`. You must **copy two files** from this `output/` folder to the main `Input Datasets/` folder at the top level of the project:
      *   `train_dataset.xlsx`
      *   `inventory_transactions_clean.xlsx` (This file is generated from the full training history and serves as the master inventory map for the dashboard).

#### Stage 2: Generate the Test Dataset
   a. **Place Raw Data:** Add the raw Excel files for your chosen test period (e.g., `Inventory Transaction Data Mar 2025.xlsx`, `Movie_sessions_Mar 2025.xlsx`) into the `Test Dataset Cleanup/input/` folder.
   b. **Run Notebook:** Open and run all cells in the `Test Dataset Cleanup/Village Roadshows - Test Dataset Cleanup.ipynb` notebook.
   c. **Action Required:** This notebook will produce the final `test_dataset.xlsx` in its `output/` folder. **Copy** this file to the main `Input Datasets/` folder at the top level.

After completing these stages, the `Input Datasets` folder should contain the three essential `.xlsx` files required to run the dashboard.

### 4. Running the Interactive Dashboard
The Streamlit application reads directly from the `Input Datasets` folder.

#### Path Configuration
> **Important:** Ensure the `TRAINING_PATH`, `HOLDOUT_PATH`, and `INVENTORY_PATH` constants at the top of `cinema_dashboard.py` point to the correct files within the `Input Datasets` folder. For example:
> `TRAINING_PATH = Path("Input Datasets/train_dataset.xlsx")`

#### Launch Command
1.  Navigate to the project's root directory in your terminal.
2.  Execute the following command:
    ```bash
    streamlit run cinema_dashboard.py
    ```
3.  The "Village Cinemas: F&B Recommender App" will open in your default web browser.
4.  The script will automatically create a `models/` directory in the project root to cache computationally expensive models and rules for faster subsequent runs.

### 5. Updating Test Data for New Periods
To evaluate the system on new data (e.g., April sales):
1.  Place the new raw data file(s) for April into the `Test Dataset Cleanup/input/` folder.
2.  Re-run the `Village Roadshows - Test Dataset Cleanup.ipynb` notebook.
3.  Copy the newly generated `test_dataset.xlsx` from its `output/` folder to the main `Input Datasets/` folder, replacing the old test file.
4.  Relaunch the Streamlit dashboard. The recommendation results will remain the same (as they are based on the same training data), but the metrics on the "Recommendation Evaluation" tab will update to reflect performance on the new April data.

---

## 📦 Installation Requirements

Install the required Python packages using:
```bash
pip install -r requirements.txt
```

---

## 👨‍💻 Authors & Project Context

This Market Basket Analysis and Recommendation Dashboard is a component of the "Village Cinema F&B Cost and Market Basket Analysis" project.

**Team:**
*   Alen George
*   Pranjal Jhala
*   Ankith Thomas
*   Aanchal
*   Abdulrahman Asiri
*   Vinit Patnaik

**Academic Supervisor:** Dr. Yameng Peng
**Industry Partner:** Village Roadshow Group Services Pty Ltd
**Course:** COSC2667/2777 – Data Science and AI Postgraduate Projects, RMIT University

