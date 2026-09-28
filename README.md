# Fake Job Posting Detector — Streamlit demo

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```
Then open the local URL Streamlit prints (usually http://localhost:8501).

## Files
- `app.py` — the Streamlit app
- `job_features.py` — feature engineering, shared with the training notebook (must match)
- `fake_job_model.joblib` — the trained model bundle (Logistic Regression + tuned threshold), produced by the notebook

All three files must stay in the same folder.
