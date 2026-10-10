# Formula 1 full-feature data collector:
# - Uses FastF1 to gather qualifying and race results for the year set in YEAR,
#   plus practice best laps, long-run pace, and available weather measurements.
# - Saves F1_<YEAR>_FULL_FEATURES.csv in the current working directory. Requires
#   internet access; FastF1 caches downloads in ML/cache.
# - Run from the repository root with: py "ML/Data collection/data2.py"
# - Requires: py -m pip install fastf1 pandas numpy

import fastf1

import pandas as pd
import numpy as np
import traceback
import time

fastf1.Cache.enable_cache("ML/cache")

YEAR = 2017
rows = []

def try_get_session(year, rnd, code):
    try:
        s = fastf1.get_session(year, rnd, code)

        try:
            s.load(telemetry=False, weather=False, messages=False)
            print(f"  {code} loaded from cache for round {rnd}")
            return s
        except:
            print(f"  {code} not cached, downloading…")

        s.load()
        time.sleep(0.1)
        return s

    except Exception as e:
        print(f"  {code} not available for round {rnd}: {e}")
        return None


def safe_results(df):
    df = df.copy()

    # Drop ALL index levels completely
    df.index = range(len(df))

    # Flatten MultiIndex columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(c) for c in col]).strip() for col in df.columns.values]

    # If DriverNumber appears multiple times, rename extras
    cols = []
    seen = set()
    for c in df.columns:
        if c == "DriverNumber":
            if c in seen:
                cols.append("DriverNumber_dup")
            else:
                cols.append(c)
                seen.add(c)
        else:
            cols.append(c)
    df.columns = cols

    return df



def get_best_lap_safe(session):
    try:
        laps = session.laps.pick_fastest()
        return laps["LapTime"].total_seconds() if laps is not None else np.nan
    except:
        return np.nan


def get_long_run_pace_safe(session):
    try:
        laps = session.laps
        laps = laps[laps["LapTime"] > pd.Timedelta("1:20")]
        if len(laps) < 5:
            return np.nan
        return laps["LapTime"].dt.total_seconds().mean()
    except:
        return np.nan


schedule = fastf1.get_event_schedule(YEAR)

for idx in schedule.index:
    try:
        event = schedule.loc[idx]
        rnd = int(event["RoundNumber"])

        print(f"\nProcessing Round {rnd} ({event['EventName']})")

        fp1 = try_get_session(YEAR, rnd, "FP1")
        fp2 = try_get_session(YEAR, rnd, "FP2")
        fp3 = try_get_session(YEAR, rnd, "FP3")

        q = try_get_session(YEAR, rnd, "Q")
        r = try_get_session(YEAR, rnd, "R")

        if q is None or r is None:
            print(f"Skipping Round {rnd}: Missing Q or R")
            continue

        # SAFE RESULTS EXTRACTION
        q_res = safe_results(q.results)[["DriverNumber", "Abbreviation", "TeamName", "Position"]]
        q_res = q_res.rename(columns={"Position": "QualifyingPosition"})

        r_res = safe_results(r.results)[["DriverNumber", "Position"]]
        r_res = r_res.rename(columns={"Position": "RacePosition"})

        merged = q_res.merge(r_res, on="DriverNumber", how="inner")

        merged["FP1_BestLap"] = get_best_lap_safe(fp1) if fp1 else np.nan
        merged["FP2_BestLap"] = get_best_lap_safe(fp2) if fp2 else np.nan
        merged["FP3_BestLap"] = get_best_lap_safe(fp3) if fp3 else np.nan
        merged["FP2_LongRunPace"] = get_long_run_pace_safe(fp2) if fp2 else np.nan

        weather_source = fp2 if fp2 else fp1

        if weather_source:
            try:
                weather = weather_source.weather_data
                merged["AirTemp"] = weather["AirTemp"].mean()
                merged["TrackTemp"] = weather["TrackTemp"].mean()
                merged["Humidity"] = weather["Humidity"].mean()
                merged["WindSpeed"] = weather["WindSpeed"].mean()
                merged["WindDirection"] = weather["WindDirection"].mean()
            except:
                merged["AirTemp"] = np.nan
                merged["TrackTemp"] = np.nan
                merged["Humidity"] = np.nan
                merged["WindSpeed"] = np.nan
                merged["WindDirection"] = np.nan
        else:
            merged["AirTemp"] = np.nan
            merged["TrackTemp"] = np.nan
            merged["Humidity"] = np.nan
            merged["WindSpeed"] = np.nan
            merged["WindDirection"] = np.nan

        merged["Circuit"] = event["EventName"]
        merged["Country"] = event["Country"]
        merged["Round"] = rnd
        merged["Year"] = YEAR

        rows.append(merged)

    except Exception as e:
        print(f"Skipping Round {idx}: {e}")
        traceback.print_exc()


if len(rows) == 0:
    print("ERROR: No data collected.")
else:
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(f"F1_{YEAR}_FULL_FEATURES.csv", index=False)
    print("\nDone — dataset created.")
