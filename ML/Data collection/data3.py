import pandas as pd
import time
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams

print("Starting NBA team game data collection...\n")

# Get all NBA teams
nba_teams = teams.get_teams()
team_ids = [t["id"] for t in nba_teams]

all_games = []

for i, team_id in enumerate(team_ids, start=1):
    print(f"[{i}/{len(team_ids)}] Fetching games for team ID {team_id}...")

    try:
        gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id)
        df_games = gamefinder.get_data_frames()[0]

        # Convert SEASON_ID to integer (e.g., '22026' → 2026)
        df_games["SEASON_ID_INT"] = df_games["SEASON_ID"].astype(str).str[-4:].astype(int)

        # Keep ONLY last 5 seasons
        df_games = df_games[df_games["SEASON_ID_INT"] >= 2020].drop(columns=["SEASON_ID_INT"])

        print(f"   → Retrieved {len(df_games)} games.")
        all_games.append(df_games)

    except Exception as e:
        print(f"   ⚠️ Error fetching games for team {team_id}: {e}")

    time.sleep(1)  # avoid rate limits

print("\nCombining all team game data...")

if len(all_games) == 0:
    print("No data collected. Something went wrong.")
else:
    games_df = pd.concat(all_games, ignore_index=True)

    # Remove duplicates (same game appears for both teams)
    games_df = games_df.drop_duplicates(subset=["GAME_ID", "TEAM_ID"])

    print(f"Total unique team-game rows: {len(games_df)}")

    output_file = "nba_team_games_last_5_years.csv"
    games_df.to_csv(output_file, index=False)

    print(f"\nDone! Data saved to: {output_file}")
