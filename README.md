# House Price Prediction using Ensemble Learning

This project predicts house prices using **ensemble learning** — instead
of relying on one model, it combines several models together so the
final prediction is more accurate.

🔗 **Live demo:** _add your Railway link here_

---

## What's Inside

```
house-price-prediction/
├── data/          # the housing dataset
├── model/         # trains the machine learning model
├── backend/       # the API that serves predictions
├── frontend/      # the webpage users interact with
└── requirements.txt
```

## How It Works

The project trains a few different models on housing data (things like
area, bedrooms, bathrooms, age, location), then combines their
predictions into one final answer. Combining models like this usually
gives a more reliable result than trusting just one model.

## How to Run It on Your Own Computer

**1. Install the required packages**
```bash
pip install -r requirements.txt
```

**2. Train the model**
```bash
python model/train_model.py
```
This creates the trained model files needed for predictions.

**3. Start the app**
```bash
cd backend
python app.py
```

**4. Open it in your browser**

Go to **http://localhost:5000** — fill in the house details and click
predict.

---

## How to Use the API Directly

Send a POST request to `/api/predict` with the house details:

```json
{
  "area_sqft": 2200,
  "bedrooms": 3,
  "bathrooms": 2,
  "stories": 2,
  "garage_spaces": 2,
  "age_years": 8,
  "location_score": 7.5,
  "has_pool": 1,
  "has_basement": 1,
  "near_main_road": 0,
  "furnishing": "furnished",
  "condition": "good"
}
```

You'll get back a predicted price in response.

---

## Deployed Online

This project is deployed on **Railway**. The live app trains itself
automatically the first time it starts, so the very first visit after
a deploy can take about 15-20 seconds to load — after that, it's fast.

---

## Built With

- **Python** + **scikit-learn** for the machine learning
- **Flask** for the backend API
- **HTML / CSS / JavaScript** for the frontend
