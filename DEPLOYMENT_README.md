# HR Attrition Analysis — Streamlit App

A complete end-to-end data analytics web app built with **Streamlit**, **Plotly**, and **Scikit-learn**.

Live demo: deploy to Streamlit Cloud in 3 minutes using the steps below.

---

## Features

| Page | What you get |
|------|-------------|
| **Overview** | KPI cards, attrition donut chart, department bar chart, tenure and age trend charts |
| **EDA** | 12+ interactive charts across demographics, job factors, compensation, and correlations |
| **ML Models** | Logistic Regression, Random Forest, Gradient Boosting — ROC curves, confusion matrix, feature importance |
| **Predict Risk** | Enter any employee's details and get an instant attrition probability from all 3 models |
| **Report** | Auto-generated executive summary with findings and recommendations + CSV download |

---

## Deploy to Streamlit Cloud (free, 3 minutes)

### Step 1 — Push to GitHub

```bash
# Create a new repo on github.com, then:
git init
git add .
git commit -m "HR Attrition Streamlit App"
git remote add origin https://github.com/YOUR_USERNAME/hr-attrition-app.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud

1. Go to **https://share.streamlit.io**
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repo → branch: `main` → Main file: `app.py`
5. Click **Deploy** — done!

Your app will be live at:
`https://YOUR_USERNAME-hr-attrition-app-app-XXXX.streamlit.app`

---

## Run locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

App opens at: **http://localhost:8501**

---

## Project structure

```
hr-attrition-app/
├── app.py                    # Main Streamlit app (all 5 pages)
├── HR_Attrition.csv          # Dataset (built-in sample or upload your own)
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml           # Theme and server config
└── README.md
```

---

## Dataset

The app ships with a built-in synthetic IBM-style HR dataset (1,470 employees).

For best results, download the **real IBM HR Analytics dataset** from Kaggle:
https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

Upload it via the sidebar file uploader in the app.

---

## Tech stack

- **Streamlit** — web app framework
- **Plotly** — interactive charts
- **Scikit-learn** — ML models (Logistic Regression, Random Forest, Gradient Boosting)
- **Pandas / NumPy** — data processing

---

*Portfolio project — Data Analytics Student Project*
