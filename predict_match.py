import pandas as pd
import joblib

from elo import update_elo

DEFAULT_ELO = 1500


def get_form(df, team):
    matches = df[
        (df["HomeTeam"] == team) |
        (df["AwayTeam"] == team)
    ].tail(10)

    wins = draws = losses = 0
    gf = ga = 0

    for _, row in matches.iterrows():

        if row["HomeTeam"] == team:
            gf += row["FTHG"]
            ga += row["FTAG"]

            if row["FTR"] == "H":
                wins += 1
            elif row["FTR"] == "D":
                draws += 1
            else:
                losses += 1
        else:
            gf += row["FTAG"]
            ga += row["FTHG"]

            if row["FTR"] == "A":
                wins += 1
            elif row["FTR"] == "D":
                draws += 1
            else:
                losses += 1

    played = max(len(matches), 1)

    return [
        wins,
        draws,
        losses,
        gf / played,
        ga / played,
        (gf - ga) / played
    ]


model = joblib.load("model.pkl")
df = pd.read_csv("data/matches.csv")

home = input("Home Team: ")
away = input("Away Team: ")

home_form = get_form(df, home)
away_form = get_form(df, away)

home_elo = DEFAULT_ELO
away_elo = DEFAULT_ELO

features = [home_form + away_form + [home_elo, away_elo]]

probs = model.predict_proba(features)[0]

print("\nPrediction")
print("-" * 30)

print(f"Away Win : {probs[0]:.2%}")
print(f"Draw     : {probs[1]:.2%}")
print(f"Home Win : {probs[2]:.2%}")