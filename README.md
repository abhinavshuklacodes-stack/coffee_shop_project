# Coffee Shop Revenue Prediction

A full-stack machine learning web application that predicts **daily total revenue**
for a coffee shop store location based on store, day of week, month, and week of year.

| Layer    | Technology                         |
|----------|------------------------------------|
| ML       | scikit-learn, numpy                |
| Backend  | Flask REST API (port 5000)         |
| Frontend | Streamlit, Plotly (port 8501)      |
| Data     | pandas, matplotlib, seaborn        |
| Report   | python-docx                        |

---

## Project Structure

```
coffee_shop_project/
├── data/
│   └── Coffee_Shop_Sales.csv        # Place dataset here
├── model/
│   ├── model.pkl                    # Trained LinearRegression artefact
│   ├── scaler.pkl                   # StandardScaler (fitted on training data)
│   ├── le_store.pkl                 # LabelEncoder for store locations
│   ├── metrics.json                 # Saved evaluation metrics
│   └── diagnostics.png              # Actual vs Predicted + Residual plots
├── backend/
│   └── app.py                       # Flask REST API
├── frontend/
│   └── ui.py                        # Streamlit UI (3 pages)
├── report_images/                   # Charts embedded in the report
├── train_model.py                   # Full ML training pipeline
├── generate_report.py               # Generates Coffee_Shop_Report.docx
├── requirements.txt                 # All Python dependencies
└── README.md                        # This file
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place the dataset
```
coffee_shop_project/data/Coffee_Shop_Sales.csv
```

### 3. Train the model
```bash
python train_model.py
```
Saves `model/model.pkl`, `model/scaler.pkl`, `model/le_store.pkl`,
`model/metrics.json`, and `model/diagnostics.png`.

### 4. Start the Flask backend (Terminal 1)
```bash
python backend/app.py
```
API runs at **http://localhost:5000**

### 5. Launch the Streamlit frontend (Terminal 2)
```bash
streamlit run frontend/ui.py
```
UI opens at **http://localhost:8501**

### 6. Generate the Word report
```bash
python generate_report.py
```
Produces **Coffee_Shop_Report.docx** in the project root.

---

## API Endpoints

| Method | Endpoint      | Description                            |
|--------|---------------|----------------------------------------|
| GET    | `/health`     | Service health check                   |
| POST   | `/predict`    | Predict daily store revenue            |
| GET    | `/dataset`    | Revenue breakdowns + sample records    |
| GET    | `/model_info` | Model coefficients + metrics           |

### POST `/predict` – example

**Request:**
```json
{
  "store_location": "Astoria",
  "day_of_week":    0,
  "month":          3,
  "week_of_year":   13
}
```

**Response:**
```json
{
  "predicted_daily_revenue":      $1119.61,
  "historical_avg_same_weekday":  $1304.93,
  "inputs": { "store_location": "Astoria", "day_name": "Monday", ... }
}
```

---

## Dataset

**Source:** [Kaggle – srisyra02/coffee-shop-sales-dataset](https://www.kaggle.com/datasets/srisyra02/coffee-shop-sales-dataset)

- **149,116** raw transactions
- **3 NYC stores** — Astoria, Hell's Kitchen, Lower Manhattan
- **Jan – Jun 2023**
- Aggregated to **543 daily store-level records** for training

---

## Model Performance

| Metric | Value    |
|--------|----------|
| R²     | 0.6942   |
| MAE    | $172.45  |
| RMSE   | $219.51  |

---

## Frontend Pages

| Page              | Description                                            |
|-------------------|--------------------------------------------------------|
| Predict Revenue   | Input form → POST /predict → gauge + store comparison  |
| Dataset Explorer  | KPIs, bar/line/pie charts, day-of-week analysis        |
| Model Insights    | R²/MAE/RMSE, coefficients bar chart, diagnostics plot  |

---

*Generated with IBM Bob  |  Coffee Shop Revenue Prediction Project*
