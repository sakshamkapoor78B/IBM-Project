# House Price Prediction using Ensemble Learning

A complete, end-to-end major project: a synthetic-but-realistic housing
dataset, an **ensemble learning** pipeline (bagging + boosting models
combined via Voting and Stacking regressors), a **Flask REST API**
backend, and a themed **HTML/CSS/JS** frontend ("The Appraisal Office")
that calls the API and visualizes how each model in the ensemble
contributes to the final prediction.

---

## 1. Project Structure

```
house-price-prediction/
├── data/
│   ├── generate_data.py      # Generates the synthetic housing dataset
│   └── housing_data.csv      # 2,500-row dataset (created on first run)
├── model/
│   ├── train_model.py        # Trains + evaluates the ensemble, saves artifacts
│   └── saved/                # Created after training:
│       ├── best_model.joblib       # Best-performing pipeline (deployed)
│       ├── base_models.joblib      # Individual base learners
│       ├── voting_model.joblib     # Voting ensemble
│       ├── stacking_model.joblib   # Stacking ensemble
│       └── metadata.json           # Metrics + feature importances
├── backend/
│   └── app.py                # Flask REST API (serves predictions + frontend)
├── frontend/
│   ├── index.html            # UI markup
│   ├── style.css             # "Blueprint / appraisal deed" design system
│   └── script.js             # Fetches from the API, renders results
├── requirements.txt
└── README.md
```

## 2. The Machine Learning Approach — Ensemble Learning

Ensemble learning combines multiple models so the group's combined
prediction is more accurate and more robust than any single model's.
This project uses **two families** of base learners, then **two ways
of combining them**:

**Base learners (the "appraisers"):**
| Model | Ensemble family | Idea |
|---|---|---|
| Random Forest | Bagging | Many decision trees on bootstrapped samples, averaged |
| Extra Trees | Bagging | Like Random Forest, with extra randomized splits |
| Gradient Boosting | Boosting | Trees built sequentially, each correcting the last one's errors |
| AdaBoost | Boosting | Sequentially reweights hard-to-predict samples |
| XGBoost *(optional)* | Boosting | Regularized, high-performance gradient boosting |

**Meta-ensembles (combine the appraisers into one verdict):**
- **Voting Regressor** — simple average of all base learners' predictions.
- **Stacking Regressor** — trains a lightweight meta-model (Ridge
  regression) on top of the base learners' predictions, learning how
  much to trust each one. This is the model `train_model.py` selects
  as the deployed model whenever it wins on held-out R².

On the bundled dataset, the **Stacking Ensemble** wins with **R² ≈ 0.93**,
outperforming every individual base learner — the expected result when
ensembling is done well.

> **Note on XGBoost:** it's listed in `requirements.txt` and used
> automatically if installed. `train_model.py` detects its absence and
> simply skips it, so the project still runs end-to-end without internet
> access or extra setup.

### The dataset
`data/generate_data.py` generates 2,500 synthetic houses using a
non-linear ground-truth formula (location premiums that scale with
size, age-based depreciation, categorical amenities, realistic noise).
This keeps the project **fully self-contained and reproducible** —
no external download required — while still giving the tree-based
ensembles genuine non-linear structure to learn. Swap in a real dataset
(e.g. a Kaggle housing CSV) by pointing `DATA_PATH` in `train_model.py`
at your file, as long as it has compatible columns.

**Features used:** `area_sqft`, `bedrooms`, `bathrooms`, `stories`,
`garage_spaces`, `age_years`, `location_score`, `has_pool`,
`has_basement`, `near_main_road`, `furnishing`, `condition`.

---

## 3. Setup & Running the Project

### Prerequisites
- Python 3.9+

### Step 1 — Install dependencies
```bash
cd house-price-prediction
pip install -r requirements.txt
```

### Step 2 — Generate the dataset (optional — auto-runs if missing)
```bash
python data/generate_data.py
```

### Step 3 — Train the ensemble model
```bash
python model/train_model.py
```
This prints a comparison table of every model's R² / MAE / RMSE, then
saves the trained pipelines + metadata into `model/saved/`.

### Step 4 — Run the backend (also serves the frontend)
```bash
cd backend
python app.py
```
Then open **http://localhost:5000** in your browser — the full app
(form + live predictions + appraiser panel + metrics) is served
directly by Flask, no separate frontend server required.

*(If you prefer to serve the frontend separately, e.g. via VS Code's
Live Server, `script.js` will automatically call
`http://localhost:5000/api` instead — just make sure the backend is
running.)*

---

## 4. API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Returns whether the model is loaded and ready |
| GET | `/api/model-info` | Returns metrics, feature importances, dropdown options |
| POST | `/api/predict` | Predicts price for one house; returns each base model's estimate too |

**Example request:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "area_sqft": 2200, "bedrooms": 3, "bathrooms": 2, "stories": 2,
    "garage_spaces": 2, "age_years": 8, "location_score": 7.5,
    "has_pool": 1, "has_basement": 1, "near_main_road": 0,
    "furnishing": "furnished", "condition": "good"
  }'
```

**Example response:**
```json
{
  "predicted_price": 410829.91,
  "best_model_used": "STACKING ENSEMBLE",
  "appraiser_estimates": {
    "Random Forest (bagging)": 402335.82,
    "Extra Trees (bagging)": 404104.69,
    "Gradient Boosting (boosting)": 408196.04,
    "AdaBoost (boosting)": 388263.48,
    "Voting Ensemble": 400725.01,
    "Stacking Ensemble": 410829.91
  },
  "spread": 22566.43
}
```

---

## 5. Frontend Design

The UI is themed as **"The Appraisal Office"** — each ensemble model is
framed as an independent appraiser filing an estimate on a case file,
with the stacking ensemble acting as the bureau's final verdict. This
makes the abstract ML concept (combining multiple models) concrete and
visual: a panel of bars showing every appraiser's number next to the
official, gold-sealed verdict.

- **Palette:** ink blue background with cyan blueprint linework, a
  cream "deed paper" card for the form and results, a stamp-red accent
  for actions/alerts, and a gold seal for the final valuation figure.
- **Type:** Fraunces (display serif) for headings, IBM Plex Sans for
  body copy, IBM Plex Mono for data/labels — evoking a ledger or deed.
- **Signature element:** the Appraiser Panel — animated bars comparing
  every base model's estimate against the final stacking verdict.

---

## 6. Deploying to Railway

This project deploys to [Railway](https://railway.app) as a single web
service — Flask serves both the API and the frontend from one process,
so there's nothing extra to configure for hosting the UI separately.

**What makes this deploy-ready:**
- `Procfile` tells Railway how to start the app with `gunicorn` (a
  production-grade server, unlike Flask's built-in dev server).
- `backend/app.py` reads the port Railway assigns via the `$PORT`
  environment variable, instead of a hardcoded port.
- The trained model files (`model/saved/*.joblib`) are **not** committed
  to git (see `.gitignore`) — they're large binary files that are easy
  to regenerate. Instead, `app.py` automatically trains the model itself
  the first time it boots with no saved model present, then loads it
  normally. This means the first request after each deploy takes about
  15–20 seconds while it trains; every request after that is instant.

### Steps

**1. Push the project to GitHub**
```bash
cd house-price-prediction
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

**2. Create the Railway project**
- Go to [railway.app](https://railway.app) and log in.
- Click **New Project → Deploy from GitHub repo**.
- Select your repository. Railway auto-detects it's a Python app
  (via `requirements.txt`) and uses Nixpacks to build it automatically.

**3. Let it deploy**
- Railway installs everything in `requirements.txt`, then runs the
  command in `Procfile`.
- Watch the **Deploy Logs** tab — you should see the training pipeline
  print its model comparison table on first boot, then
  `Loaded model artifacts from ...` once it's ready.

**4. Get your public URL**
- In your service's **Settings → Networking**, click **Generate Domain**.
- Railway gives you a URL like `https://your-app.up.railway.app` —
  open it and the full app (form, predictions, appraiser panel) works
  exactly like it did locally.

### Notes & tradeoffs
- **Cold start on every deploy/restart:** since the trained model isn't
  persisted, a new deploy or a container restart means retraining once
  again on the next request (~15–20s, one time). For a demo/portfolio
  project this is a fine tradeoff for a much smaller, simpler repo. If
  you want to avoid this:
  - Add a [Railway Volume](https://docs.railway.app/reference/volumes)
    mounted at `model/saved/` so the trained files persist across
    deploys, or
  - Remove `model/saved/*.joblib` from `.gitignore` and commit the
    trained files directly (they're roughly 50–70MB total after
    training once locally with `python model/train_model.py`).
- **Single worker:** the `Procfile` runs gunicorn with `--workers 1`.
  Since the model is loaded into memory per-worker, more workers means
  more memory usage and multiple independent training runs on cold
  boot. One worker is enough for a demo; scale up only if you add
  persistence first.
- **Logs are your friend:** if `/api/health` ever reports
  `model_loaded: false`, check the Railway deploy logs — the training
  step logs a full error if something goes wrong (e.g. an out-of-memory
  error on a very small instance size, in which case increase your
  Railway plan's memory or reduce `n_estimators` in `train_model.py`).

## 7. Extending the Project

- Swap the synthetic dataset for a real one (e.g. Kaggle's "House
  Prices — Advanced Regression Techniques") by adjusting the column
  names in `train_model.py`.
- Add hyperparameter tuning (`GridSearchCV` / `RandomizedSearchCV`) per
  base learner before ensembling.
- Add a `/api/batch-predict` endpoint for CSV uploads.
- Containerize with Docker (`Dockerfile` for backend + a static file
  server, or serve the frontend from Flask as already configured).
- Deploy the backend (e.g. Render, Railway, EC2) and point the
  frontend's `API_BASE` at the deployed URL.
