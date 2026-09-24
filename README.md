# Car Price Predictor (Streamlit)

## Run locally
1. Install dependencies:
   pip install -r requirements.txt

2. Make sure these files are in the same folder as app.py:
   - car_price_dataset.csv
   - ColumnTransformer.pkl
   - model.keras

3. Run:
   streamlit run app.py

Note: ColumnTransformer.pkl was saved with joblib, so it must be loaded with
joblib.load (already handled in app.py) — plain pickle.load will fail.
