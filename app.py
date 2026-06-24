import gradio as gr
import pandas as pd
import joblib
import os

from features import team_form

# -----------------------------
# LOAD MODEL + DATA
# -----------------------------
BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
df = pd.read_csv(os.path.join(BASE_DIR, "matches.csv"))

teams = sorted(df["HomeTeam"].unique())

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
def build_features(team):
    return team_form(df, team)

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict(home, away):

    home_form = build_features(home)
    away_form = build_features(away)

    home_elo = 1500
    away_elo = 1500

    X = [home_form + away_form + [home_elo, away_elo]]

    probs = model.predict_proba(X)[0]

    return {
        "Home Win": float(probs[2]),
        "Draw": float(probs[1]),
        "Away Win": float(probs[0])
    }

# -----------------------------
# UI
# -----------------------------
app = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(teams, label="Home Team"),
        gr.Dropdown(teams, label="Away Team")
    ],
    outputs="json",
    title="⚽ Football Match Predictor AI",
    description="Predict Premier League match outcomes using ML + form stats"
)

app.launch()