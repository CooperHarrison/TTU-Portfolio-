import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error



df = pd.read_csv('ML//F1 data//final_dataset.csv')
df = pd.get_dummies(df)
df = df.dropna()

X = df.drop(['RacePosition'], axis=1)
y = df['RacePosition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)
rfr = RandomForestRegressor(random_state=13)
rfr.fit(X_train, y_train)
y_pred = rfr.predict(X_test)

def predict_position(Abbreviation, TeamName, QualifyingPosition, Circuit, Year):
    
    sample = pd.DataFrame(columns=X.columns)
    sample.loc[0] = 0  

    if 'QualifyingPosition' in sample.columns:
        sample.loc[0, 'QualifyingPosition'] = QualifyingPosition

  
    abb_col = f"Abbreviation_{Abbreviation}"
    team_col = f"TeamName_{TeamName}"
    circuit_col = f"Circuit_{Circuit}"
    year_col = f"Year_{Year}"

    for col in [abb_col, team_col, circuit_col, year_col]:
        if col in sample.columns:
            sample.loc[0, col] = 1
        else:
            print(f"Warning: column '{col}' not found in training data")

    prediction = rfr.predict(sample)[0]
    return prediction



print(predict_position(
    Abbreviation="LEC",
    TeamName="Ferrari",
    QualifyingPosition= 4,
    Circuit="Chinese Grand Prix",
    Year=2026
))

print(mean_absolute_error(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))