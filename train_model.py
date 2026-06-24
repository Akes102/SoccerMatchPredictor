import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("matches.csv")
df = df.dropna()

# Features
X = df[["FTHG", "FTAG"]]

# Target
y = df["FTR"].map({
    "H": 2,
    "D": 1,
    "A": 0
})

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

joblib.dump(model, "model.pkl")

print("Model trained successfully")