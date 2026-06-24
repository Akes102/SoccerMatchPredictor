import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# LOAD DATA
df = pd.read_csv("matches.csv")

# FEATURES
def build_features(row):
    return [
        row["FTHG"],
        row["FTAG"],
    ]

X = []
y = []

for _, r in df.iterrows():

    X.append([
        r["FTHG"],
        r["FTAG"]
    ])

    # encode result
    if r["FTR"] == "H":
        y.append(2)
    elif r["FTR"] == "D":
        y.append(1)
    else:
        y.append(0)

# MODEL
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# SAVE CLEAN MODEL
joblib.dump(model, "model.pkl")

print("Model trained and saved cleanly")