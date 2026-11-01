# WeathOut (Weather + Outfit)
# Main idea: you input a temperature (температура), a wind speed (ветер), a rain and a  humidity (влажность). Program outputs clothes for this weather.
# Stack: Python, Scikit-learn, Pandas

from sklearn import tree
import joblib
from data import get_csv, code_csv
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"

df = code_csv(get_csv())
X = df[["temperature", "wind_speed", "humidity", "rain"]]
y = df["outfit"]

clf = tree.DecisionTreeClassifier(max_depth=5, random_state=1)
clf = clf.fit(X, y)

joblib.dump(clf, MODEL_PATH)