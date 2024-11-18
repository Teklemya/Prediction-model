import { useState, useEffect } from 'react';
import axios from 'axios';

const OddsFetcher = () => {
  const [odds, setOdds] = useState([]);
  const [sport, setSport] = useState('americanfootball_nfl');
  const [prediction, setPrediction] = useState('');

  // Fetch odds data
  useEffect(() => {
    axios
      .get(`/odds?sport=${sport}`)
      .then((response) => {
        setOdds(response.data);
      })
      .catch((error) => {
        console.error('Error fetching odds:', error);
      });
  }, [sport]);

  // Handle prediction requests
  const handlePrediction = (team1Odds, team2Odds) => {
    axios
      .post('/predict', {
        team_1_odds: team1Odds,
        team_2_odds: team2Odds,
      })
      .then((response) => {
        setPrediction(response.data.prediction);
      })
      .catch((error) => {
        console.error('Error making prediction:', error);
      });
  };

  return (
    <div>
      <h1>American Football Odds</h1>
      <select onChange={(e) => setSport(e.target.value)} value={sport}>
        <option value="americanfootball_nfl">NFL</option>
        <option value="americanfootball_ncaaf">College Football</option>
      </select>
      <div>
        {odds.length > 0 ? (
          odds.map((game, index) => (
            <div key={index}>
              <h2>
                {game.team_1} vs {game.team_2}
              </h2>
              <p>Team 1 Odds: {game.team_1_odds}</p>
              <p>Team 2 Odds: {game.team_2_odds}</p>
              <button
                onClick={() =>
                  handlePrediction(game.team_1_odds, game.team_2_odds)
                }
              >
                Predict Outcome
              </button>
            </div>
          ))
        ) : (
          <p>No odds available.</p>
        )}
      </div>
      {prediction && <h3>Prediction: {prediction}</h3>}
    </div>
  );
};

export default OddsFetcher;
