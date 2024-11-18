from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)
CORS(app)

# Your OddsAPI key
ODDS_API_KEY = "2e7c28fce575b5167c9353a678f465dc"
BASE_URL = "https://api.the-odds-api.com/v4/sports/"

# Example training data for American Football
# Replace with historical data for better predictions
def get_training_data():
    data = {
        "team_1_odds": [-200, -150, +120, +300, -110],
        "team_2_odds": [+180, +130, -140, -400, -105],
        "outcome": [1, 1, 0, 0, 1]  # 1: Team 1 wins, 0: Team 2 wins
    }
    return pd.DataFrame(data)

# Train the model
def train_model():
    df = get_training_data()
    X = df[["team_1_odds", "team_2_odds"]]
    y = df["outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, predictions):.2f}")
    return model

model = train_model()

# Fetch odds data
@app.route('/odds', methods=['GET'])
def get_odds():
    sport = request.args.get('sport', 'americanfootball_nfl')  # Default to NFL
    url = f"{BASE_URL}{sport}/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h"  # Head-to-Head markets
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        odds_data = response.json()
        
        # Process odds data
        processed_data = []
        for game in odds_data:
            if "bookmakers" not in game or not game["bookmakers"]:
                continue
            outcomes = game["bookmakers"][0]["markets"][0]["outcomes"]
            processed_data.append({
                "team_1": outcomes[0]["name"],
                "team_2": outcomes[1]["name"],
                "team_1_odds": outcomes[0]["price"],  # American odds
                "team_2_odds": outcomes[1]["price"]
            })
        return jsonify(processed_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Predict game outcome
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    team_1_odds = data.get("team_1_odds")
    team_2_odds = data.get("team_2_odds")
    
    if team_1_odds is None or team_2_odds is None:
        return jsonify({"error": "Invalid input"}), 400
    
    prediction = model.predict([[team_1_odds, team_2_odds]])
    result = "Team 1 Wins" if prediction[0] == 1 else "Team 2 Wins"
    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(debug=True)
