import gradio as gr
import pandas as pd
import joblib
import os

BASE = os.path.dirname(__file__)

df = pd.read_csv(os.path.join(BASE, "matches.csv"))
model = joblib.load(os.path.join(BASE, "model.pkl"))

teams = sorted(df["HomeTeam"].unique())

def predict(home, away):

    # SIMPLE CONSISTENT FEATURES ONLY
    # (NO FORM, NO ELO — must match training)

    home_goals = df[df["HomeTeam"] == home]["FTHG"].mean()
    home_concede = df[df["HomeTeam"] == home]["FTAG"].mean()

    away_goals = df[df["AwayTeam"] == away]["FTAG"].mean()
    away_concede = df[df["AwayTeam"] == away]["FTHG"].mean()

    # fallback safety
    home_goals = 0 if pd.isna(home_goals) else home_goals
    home_concede = 0 if pd.isna(home_concede) else home_concede
    away_goals = 0 if pd.isna(away_goals) else away_goals
    away_concede = 0 if pd.isna(away_concede) else away_concede

    X = [[
        home_goals,
        home_concede
    ]]

    probs = model.predict_proba(X)[0]
    probs = probs / sum(probs)

    return {
        "Home Win %": f"{round(probs[2]*100,1)}%",
        "Draw %": f"{round(probs[1]*100,1)}%",
        "Away Win %": f"{round(probs[0]*100,1)}%"
    }

app = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(teams),
        gr.Dropdown(teams)
    ],
    outputs="json",
    title="Football AI Predictor"
)

app.launch()