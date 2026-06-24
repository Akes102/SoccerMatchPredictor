import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

from elo import update_elo
from features import team_form

DEFAULT_ELO = 1500

df = pd.read_csv("data/matches.csv")

elo = {}

X = []
y = []

for i in range(len(df)):

    home = df.iloc[i]["HomeTeam"]
    away = df.iloc[i]["AwayTeam"]

    home_form = team_form(df.iloc[:i], home)
    away_form = team_form(df.iloc[:i], away)

    home_elo = elo.get(home, DEFAULT_ELO)
    away_elo = elo.get(away, DEFAULT_ELO)

    X.append(home_form + away_form + [home_elo, away_elo])

    result = df.iloc[i]["FTR"]

    if result == "H":
        label = 2
        score = 1
    elif result == "D":
        label = 1
        score = 0.5
    else:
        label = 0
        score = 0

    y.append(label)

    elo[home], elo[away] = update_elo(home_elo, away_elo, score)

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    random_state=42
)

model.fit(X, y)

joblib.dump(model, "model.pkl")

print("PRO MODEL TRAINED")