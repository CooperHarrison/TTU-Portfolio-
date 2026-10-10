# Formula 1 race-position model (CatBoost):
# - Reads ML/F1 data/final_dataset2.csv and trains a regressor using qualifying,
#   practice, weather, circuit, and other race features.
# - Prints a sample prediction and evaluation metrics for the held-out test set.
# - Run from the repository root with: py ML/Models/Cboost.py
# - Requires: py -m pip install pandas scikit-learn catboost

import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


df = pd.read_csv('ML//F1 data//final_dataset2.csv', skip_blank_lines=True, encoding='utf-8-sig')

df = df[df["RacePosition"].notna()]

df = df.fillna(-1)

y = df["RacePosition"]
X = df.drop(columns=["RacePosition"])



X = df.drop(['RacePosition', 'DriverNumber', 'FP2_LongRunPace', 'Round'], axis=1)
y = df['RacePosition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)

model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.01,
    depth=10,
    loss_function='RMSE',
    verbose=False
)
model.fit(X_train, y_train, cat_features=['Abbreviation', 'TeamName', 'Circuit', 'Country', 'Year'])

y_pred = model.predict(X_test)


def predict_position(sample_dict):
    sample = pd.DataFrame([sample_dict])
    return model.predict(sample)[0]


sample = {
    "Abbreviation": "VER",
    "TeamName": "Red Bull Racing",
    "QualifyingPosition": 11,
    "FP1_BestLap": 91.666,
    "FP2_BestLap": 90.133,
    "FP3_BestLap": 89.362,
    "AirTemp": 17.171764705882353,
    "TrackTemp": 28.64588235294117,
    "Humidity": 46.41411764705882,
    "WindSpeed": 1.8152941176470596,
    "WindDirection": 126.85882352941177,
    "Circuit": "Chinese Grand Prix",
    "Country": "China",
    "Year": 2018
}

predicted_position = predict_position(sample)
print("Predicted RacePosition:", predicted_position)
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

