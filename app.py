import streamlit as st
import pandas as pd
import joblib
import os

from features import team_form

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Pro Football AI", layout="wide")
st.title("⚽ Pro Football Prediction Engine")

# -----------------------------
# SAFE PATHS (IMPORTANT FIX)
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "matches.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# -----------------------------
# LOAD MODEL (SAFE)
# -----------------------------
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()

# -----------------------------
# LOAD DATA (FIXED PATH)
# -----------------------------
try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error(f"Dataset load failed: {e}")
    st.stop()

# -----------------------------
# TEAM LIST
# -----------------------------
teams = sorted(df["HomeTeam"].unique())

col1, col2 = st.columns(2)

with col1:
    home = st.selectbox("Home Team", teams)

with col2:
    away = st.selectbox("Away Team", teams)

st.markdown("---")

# -----------------------------
# PREDICTION LOGIC
# -----------------------------
def build_features(df, team):
    return team_form(df, team)

if st.button("Run Prediction"):

    home_form = build_features(df, home)
    away_form = build_features(df, away)

    # placeholder Elo (you can upgrade later)
    home_elo = 1500
    away_elo = 1500

    X = [home_form + away_form + [home_elo, away_elo]]

    probs = model.predict_proba(X)[0]

    home_win = probs[2]
    draw = probs[1]
    away_win = probs[0]

    # -----------------------------
    # UI OUTPUT
    # -----------------------------
    st.subheader("Match Probability")

    col1, col2, col3 = st.columns(3)

    col1.metric("Home Win", f"{home_win:.2%}")
    col2.metric("Draw", f"{draw:.2%}")
    col3.metric("Away Win", f"{away_win:.2%}")

    st.progress(float(home_win))

    st.markdown("---")

    st.subheader("AI Insight")

    if home_win > 0.55:
        st.write(f"{home} is strongly favoured at home.")
    elif away_win > 0.55:
        st.write(f"{away} has the upper hand away from home.")
    else:
        st.write("Balanced match. Draw is realistic.")