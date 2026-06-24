import pandas as pd
import joblib
import os
from features import team_form
from elo import update_elo

# -----------------------------
# LOAD DATA + MODEL
# -----------------------------
BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "matches.csv")

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

teams = sorted(df["HomeTeam"].unique())

# -----------------------------
# BUILD FEATURE VECTOR
# -----------------------------
def build_features(team):
    return team_form(df, team)

# -----------------------------
# PREDICT FUNCTION
# -----------------------------
def predict_match(home, away):

    home_form = build_features(home)
    away_form = build_features(away)

    # Elo memory store
    if not hasattr(predict_match, "elo"):
        predict_match.elo = {t: 1500 for t in teams}

    home_elo = predict_match.elo.get(home, 1500)
    away_elo = predict_match.elo.get(away, 1500)

    # model input
    X = [home_form + away_form + [home_elo, away_elo]]

    probs = model.predict_proba(X)[0]

    # convert to match result strength
    result = probs[2] + 0.5 * probs[1]

    # Elo update
    expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))

    new_home_elo, new_away_elo = update_elo(
        home_elo,
        away_elo,
        result
    )

    predict_match.elo[home] = new_home_elo
    predict_match.elo[away] = new_away_elo

    # normalize probabilities
    probs = probs / probs.sum()

    return {
        "Home Win (%)": f"{round(probs[2] * 100, 1)}%",
        "Draw (%)": f"{round(probs[1] * 100, 1)}%",
        "Away Win (%)": f"{round(probs[0] * 100, 1)}%"
    }

# -----------------------------
# TEST RUN (optional local use)
# -----------------------------
if __name__ == "__main__":
    home = input("Home team: ")
    away = input("Away team: ")

    result = predict_match(home, away)
    print(result)