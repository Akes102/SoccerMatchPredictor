import gradio as gr
import pandas as pd
import os
import joblib

from features import team_form
from elo_engine import EloEngine

# -----------------------------
# LOAD DATA
# -----------------------------
BASE = os.path.dirname(__file__)

df = pd.read_csv(os.path.join(BASE, "matches.csv"))
model = joblib.load(os.path.join(BASE, "model.pkl"))

teams = sorted(df["HomeTeam"].unique())

elo = EloEngine(teams)

# -----------------------------
# FEATURE BUILDER
# -----------------------------
def build(team):
    return team_form(df, team)

# -----------------------------
# PREDICTION ENGINE
# -----------------------------
def predict(home, away):

    home_form = build(home)
    away_form = build(away)

    home_elo = elo.ratings[home]
    away_elo = elo.ratings[away]

    X = [home_form + away_form + [home_elo, away_elo]]

    probs = model.predict_proba(X)[0]
    probs = probs / probs.sum()

    # convert to match strength
    result_strength = probs[2] + 0.5 * probs[1]

    elo.update(home, away, result_strength)

    return {
        "Home Win (%)": f"{round(probs[2] * 100, 1)}%",
        "Draw (%)": f"{round(probs[1] * 100, 1)}%",
        "Away Win (%)": f"{round(probs[0] * 100, 1)}%",
        "Home Elo": round(elo.ratings[home], 1),
        "Away Elo": round(elo.ratings[away], 1)
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
    title="⚽ Pro Football AI System",
    description="Elo + Form + ML probability engine"
)

app.launch()