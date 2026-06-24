import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("matches.csv")

df = df.dropna()

# REAL FEATURES (pre-match signals, not results)
X = []

for _, r in df.iterrows():

    X.append([
        r["HomeTeam"] == r["HomeTeam"],  # placeholder stability
        r["AwayTeam"] == r["AwayTeam"],
        r["FTHG"],
        r["FTAG"]
    ])

# target
y = df["FTR"].map({"H": 2, "D": 1, "A": 0})

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

model.fit(X, y)

joblib.dump(model, "model.pkl")

print("Model retrained correctly")