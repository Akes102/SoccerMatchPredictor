import streamlit as st
import pandas as pd
import joblib

from features import team_form

st.set_page_config(page_title="Pro Football AI", layout="wide")

st.title("⚽ Pro Football Prediction Engine")

model = joblib.load("model.pkl")
df = pd.read_csv("data/matches.csv")

teams = sorted(df["HomeTeam"].unique())

col1, col2 = st.columns(2)

with col1:
    home = st.selectbox("Home Team", teams)

with col2:
    away = st.selectbox("Away Team", teams)

st.markdown("---")

if st.button("Run Prediction"):

    home_form = team_form(df, home)
    away_form = team_form(df, away)

    home_elo = 1500
    away_elo = 1500

    X = [home_form + away_form + [home_elo, away_elo]]

    probs = model.predict_proba(X)[0]

    home_win = probs[2]
    draw = probs[1]
    away_win = probs[0]

    st.subheader("Match Probability")

    col1, col2, col3 = st.columns(3)

    col1.metric("Home Win", f"{home_win:.2%}")
    col2.metric("Draw", f"{draw:.2%}")
    col3.metric("Away Win", f"{away_win:.2%}")

    st.progress(float(home_win))

    st.markdown("---")

    st.subheader("AI Insight")

    if home_win > 0.5:
        st.write(f"{home} has strong home advantage and better recent form.")
    elif away_win > 0.5:
        st.write(f"{away} is statistically stronger going into this match.")
    else:
        st.write("Match is balanced. Draw probability is significant.")