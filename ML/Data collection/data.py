import fastf1
import pandas as pd

fastf1.Cache.enable_cache("ML\\cache")   

YEAR = 2026   

rows = []


schedule = fastf1.get_event_schedule(YEAR)
total_rounds = schedule.index.max()

for rnd in range(1, total_rounds + 1):
    try:
        
        q = fastf1.get_session(YEAR, rnd, "Q")
        q.load()
        q_res = q.results[["DriverNumber", "Abbreviation", "TeamName", "Position"]]
        q_res = q_res.rename(columns={"Position": "QualifyingPosition"})

       
        r = fastf1.get_session(YEAR, rnd, "R")
        r.load()
        r_res = r.results[["DriverNumber", "Position"]]
        r_res = r_res.rename(columns={"Position": "RacePosition"})

       
        merged = q_res.merge(r_res, on="DriverNumber", how="inner")

      
        merged["Circuit"] = r.event["EventName"]
        merged["Year"] = YEAR

        rows.append(merged)
        print(f"Processed Round {rnd}")

    except Exception as e:
        print(f"Skipping Round {rnd}: {e}")

df = pd.concat(rows, ignore_index=True)
df.to_csv(f"F1_{YEAR}_DRIVER_RESULTS.csv", index=False)

print("Done — dataset created.")
