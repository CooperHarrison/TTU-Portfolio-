import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv("ML/Basketball data/nba_data.csv")

df = df[df["WL"].notna()].copy()
df["WL"] = df["WL"].apply(lambda x: 1 if x == "W" else 0)
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
df["SEASON_YEAR"] = df["SEASON_ID"].astype(str).str[-4:].astype(int)


def get_opp_abbr(matchup):
    m = matchup.replace(".", "")
    parts = m.split()
    return parts[-1]

df["OPP_ABBREVIATION"] = df["MATCHUP"].apply(get_opp_abbr)


df = df.sort_values(["TEAM_ID", "GAME_DATE"])


df["TEAM_WINS_BEFORE"] = df.groupby("TEAM_ID")["WL"].cumsum() - df["WL"]
df["TEAM_GAMES_BEFORE"] = df.groupby("TEAM_ID").cumcount()
df["TEAM_WIN_PCT_BEFORE"] = df["TEAM_WINS_BEFORE"] / df["TEAM_GAMES_BEFORE"].replace(0, 1)


df["TEAM_LAST5"] = (
    df.groupby("TEAM_ID")["WL"]
    .rolling(5, min_periods=1)
    .sum()
    .shift(1)
    .reset_index(level=0, drop=True)
).fillna(0)


opp = df[[
    "GAME_ID",
    "TEAM_ID",
    "TEAM_WIN_PCT_BEFORE",
    "TEAM_LAST5",
    "TEAM_ABBREVIATION"
]].rename(columns={
    "TEAM_ID": "OPP_TEAM_ID",
    "TEAM_WIN_PCT_BEFORE": "OPP_WIN_PCT_BEFORE",
    "TEAM_LAST5": "OPP_LAST5",
    "TEAM_ABBREVIATION": "OPP_ABBREVIATION_REAL"
})

df = df.merge(opp, on="GAME_ID", how="left")
df = df[df["TEAM_ID"] != df["OPP_TEAM_ID"]]


df["IS_HOME"] = df["MATCHUP"].str.contains("vs").astype(int)


X = df[[
    "SEASON_YEAR",
    "TEAM_WIN_PCT_BEFORE",
    "OPP_WIN_PCT_BEFORE",
    "TEAM_LAST5",
    "OPP_LAST5",
    "IS_HOME",
    "TEAM_ABBREVIATION",
    "OPP_ABBREVIATION"
]]

y = df["WL"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = CatBoostClassifier(
    iterations=1500,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=4,
    random_strength=1.5,
    bagging_temperature=0.8,
    verbose=False
)

model.fit(
    X_train, y_train,
    cat_features=["TEAM_ABBREVIATION", "OPP_ABBREVIATION"]
)


y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))




latest_season = df["SEASON_YEAR"].max()


team_stats = df.groupby("TEAM_ABBREVIATION")[[
    "TEAM_WIN_PCT_BEFORE", "TEAM_LAST5"
]].mean()

spurs_vs_knicks = pd.DataFrame([{
    "SEASON_YEAR": latest_season,
    "TEAM_WIN_PCT_BEFORE": team_stats.loc["SAS", "TEAM_WIN_PCT_BEFORE"],
    "OPP_WIN_PCT_BEFORE": team_stats.loc["NYK", "TEAM_WIN_PCT_BEFORE"],
    "TEAM_LAST5": team_stats.loc["SAS", "TEAM_LAST5"],
    "OPP_LAST5": team_stats.loc["NYK", "TEAM_LAST5"],
    "IS_HOME": 1,  
    "TEAM_ABBREVIATION": "SAS",
    "OPP_ABBREVIATION": "NYK"
}])


proba = model.predict_proba(spurs_vs_knicks)[0][1]
print(f"\nPredicted probability Spurs beat Knicks: {proba:.3f}")
