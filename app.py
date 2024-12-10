import sys
import os
from sklearn.model_selection import train_test_split

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Backend.data_loader import load_data
from Backend.feature_engineering import create_features
from Backend.model import prepare_data, train_model, evaluate_model, get_feature_importance

def main():
    try:
        # Load data
        print("Loading data...")
        df = load_data('data/nfl/spreadspoke_scores.csv')
        
        # Engineer features
        print("Engineering features...")
        df_processed = create_features(df)
        
        # Prepare data for modeling
        print("Preparing data for modeling...")
        X, y = prepare_data(df_processed)
        
        # Split data
        print("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        print("Training model...")
        model = train_model(X_train, y_train)
        
        # Evaluate model
        print("Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)
        
        # Get feature importance
        print("\nAnalyzing feature importance...")
        feature_names = X.columns.tolist()
        feature_imp = get_feature_importance(model, feature_names)
        
        print("\nModel training and evaluation complete.")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    
    
    # from flask import Flask, jsonify, request
# from flask_cors import CORS
# import requests
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# app = Flask(__name__)
# CORS(app)

# # Your OddsAPI key
# ODDS_API_KEY = "2e7c28fce575b5167c9353a678f465dc"
# BASE_URL = "https://api.the-odds-api.com/v4/sports/"

# # Example training data for American Football
# # Replace with historical data for better predictions
# def get_training_data():
#     data = {
#         "team_1_odds": [-200, -150, +120, +300, -110],
#         "team_2_odds": [+180, +130, -140, -400, -105],
#         "outcome": [1, 1, 0, 0, 1]  # 1: Team 1 wins, 0: Team 2 wins
#     }
#     return pd.DataFrame(data)

# # Train the model
# def train_model():
#     df = get_training_data()
#     X = df[["team_1_odds", "team_2_odds"]]
#     y = df["outcome"]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     model = RandomForestClassifier(random_state=42)
#     model.fit(X_train, y_train)
    
#     predictions = model.predict(X_test)
#     print(f"Model Accuracy: {accuracy_score(y_test, predictions):.2f}")
#     return model

# model = train_model()

# # Fetch odds data
# @app.route('/odds', methods=['GET'])
# def get_odds():
#     sport = request.args.get('sport', 'americanfootball_nfl')  # Default to NFL
#     url = f"{BASE_URL}{sport}/odds/"
#     params = {
#         "apiKey": ODDS_API_KEY,
#         "regions": "us",
#         "markets": "h2h"  # Head-to-Head markets
#     }
    
#     try:
#         response = requests.get(url, params=params)
#         response.raise_for_status()
#         odds_data = response.json()
        
#         # Process odds data
#         processed_data = []
#         for game in odds_data:
#             if "bookmakers" not in game or not game["bookmakers"]:
#                 continue
#             outcomes = game["bookmakers"][0]["markets"][0]["outcomes"]
#             processed_data.append({
#                 "team_1": outcomes[0]["name"],
#                 "team_2": outcomes[1]["name"],
#                 "team_1_odds": outcomes[0]["price"],  # American odds
#                 "team_2_odds": outcomes[1]["price"]
#             })
#         return jsonify(processed_data)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# # Predict game outcome
# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json
#     team_1_odds = data.get("team_1_odds")
#     team_2_odds = data.get("team_2_odds")
    
#     if team_1_odds is None or team_2_odds is None:
#         return jsonify({"error": "Invalid input"}), 400
    
#     prediction = model.predict([[team_1_odds, team_2_odds]])
#     result = "Team 1 Wins" if prediction[0] == 1 else "Team 2 Wins"
#     return jsonify({"prediction": result})

# if __name__ == '__main__':
#     app.run(debug=True)
