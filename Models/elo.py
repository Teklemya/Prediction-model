import pandas as pd
import os

# Function to calculate expected score
def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

# Function to update Elo ratings
def update_ratings(rating_a, rating_b, score_a, k=20):
    exp_a = expected_score(rating_a, rating_b)
    new_rating_a = rating_a + k * (score_a - exp_a)
    new_rating_b = rating_b + k * ((1 - score_a) - (1 - exp_a))
    return new_rating_a, new_rating_b

# Process NBA data
def process_nba_data(base_path, initial_rating=1500):
    # Relevant NBA file
    team_totals_file = os.path.join(base_path, "Team Totals.csv")
    if not os.path.exists(team_totals_file):
        print(f"File not found: {team_totals_file}")
        return

    team_totals = pd.read_csv(team_totals_file)
    
    # Initialize ratings for each team
    teams = team_totals['Team'].unique()
    ratings = {team: initial_rating for team in teams}

    # Example match simulation: Replace this with actual game data
    for _, row in team_totals.iterrows():
        team_a, team_b = row['Team'], row['Opponent']
        score_a = 1 if row['Win'] else 0  # Adjust based on data

        ratings[team_a], ratings[team_b] = update_ratings(
            ratings[team_a], ratings[team_b], score_a
        )

    # Save ratings
    ratings_df = pd.DataFrame(list(ratings.items()), columns=['Team', 'Rating'])
    ratings_df.to_excel(os.path.join(base_path, "nba_elo_ratings.xlsx"), index=False)
    print("NBA Elo ratings saved.")

# Process NFL data
def process_nfl_data(base_path, initial_rating=1500):
    # Relevant NFL file
    spread_scores_file = os.path.join(base_path, "spreadspoke_scores.csv")
    if not os.path.exists(spread_scores_file):
        print(f"File not found: {spread_scores_file}")
        return

    spread_scores = pd.read_csv(spread_scores_file)

    # Initialize ratings for each team
    teams = spread_scores['team_home'].unique()
    ratings = {team: initial_rating for team in teams}

    # Example match simulation
    for _, row in spread_scores.iterrows():
        team_a, team_b = row['team_home'], row['team_away']
        score_a = 1 if row['home_score'] > row['away_score'] else 0

        ratings[team_a], ratings[team_b] = update_ratings(
            ratings[team_a], ratings[team_b], score_a
        )

    # Save ratings
    ratings_df = pd.DataFrame(list(ratings.items()), columns=['Team', 'Rating'])
    ratings_df.to_excel(os.path.join(base_path, "nfl_elo_ratings.xlsx"), index=False)
    print("NFL Elo ratings saved.")

# Process UFC data
def process_ufc_data(base_path, initial_rating=1500):
    # Relevant UFC file
    fights_file = os.path.join(base_path, "ufcfights10_26_24.csv")
    if not os.path.exists(fights_file):
        print(f"File not found: {fights_file}")
        return

    fights = pd.read_csv(fights_file)

    # Initialize ratings for each fighter
    fighters = fights['fighter'].unique()
    ratings = {fighter: initial_rating for fighter in fighters}

    # Example match simulation
    for _, row in fights.iterrows():
        fighter_a, fighter_b = row['fighter'], row['opponent']
        score_a = 1 if row['result'] == 'win' else 0

        ratings[fighter_a], ratings[fighter_b] = update_ratings(
            ratings[fighter_a], ratings[fighter_b], score_a
        )

    # Save ratings
    ratings_df = pd.DataFrame(list(ratings.items()), columns=['Fighter', 'Rating'])
    ratings_df.to_excel(os.path.join(base_path, "ufc_elo_ratings.xlsx"), index=False)
    print("UFC Elo ratings saved.")

# Main function
def main():
    nba_path = r"C:\Users\Eyuel\Desktop\Prediction-model\data\nba"
    nfl_path = r"C:\Users\Eyuel\Desktop\Prediction-model\data\nfl"
    ufc_path = r"C:\Users\Eyuel\Desktop\Prediction-model\data\ufc"

    print("Processing NBA data...")
    process_nba_data(nba_path)

    print("Processing NFL data...")
    process_nfl_data(nfl_path)

    print("Processing UFC data...")
    process_ufc_data(ufc_path)

if __name__ == "__main__":
    main()
