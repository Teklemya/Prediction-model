import numpy as np
import pandas as pd

# 1. Kelly Criterion
def kelly_criterion(prob, odds):
    """
    Calculate the optimal bet fraction using the Kelly Criterion.

    Parameters:
        prob (float): Probability of the event.
        odds (float): Decimal odds offered by the bookmaker.

    Returns:
        float: Fraction of bankroll to bet.
    """
    b = odds - 1
    q = 1 - prob
    kelly_fraction = (b * prob - q) / b
    return max(kelly_fraction, 0)  # Ensure non-negative fraction


# 2. Portfolio Hedging
def portfolio_hedging(initial_bets, current_odds_list, model_probs_list, total_stake=1000):
    """
    Manage a portfolio of bets by determining optimal hedging strategies.

    Parameters:
        initial_bets (list): List of dictionaries, each representing a bet with 'outcome', 'odds', and 'stake'.
        current_odds_list (list): List of dictionaries, each with current odds for each outcome.
        model_probs_list (list): List of dictionaries, each with model-predicted probabilities for each outcome.
        total_stake (float): Total stake amount.

    Returns:
        pd.DataFrame: Summary of hedging recommendations.
    """
    recommendations = []

    for i, bet in enumerate(initial_bets):
        current_odds = current_odds_list[i]
        model_probs = model_probs_list[i]
        outcome = bet["outcome"]
        hedge_outcomes = [o for o in current_odds.keys() if o != outcome]

        for hedge_outcome in hedge_outcomes:
            hedge_prob = model_probs.get(hedge_outcome, 0)
            hedge_odds = current_odds.get(hedge_outcome, 0)

            if hedge_odds > 0 and hedge_prob > 0:
                hedge_stake = kelly_criterion(hedge_prob, hedge_odds) * total_stake
                recommendations.append({
                    "initial_outcome": outcome,
                    "hedge_outcome": hedge_outcome,
                    "hedge_odds": hedge_odds,
                    "hedge_prob": hedge_prob,
                    "hedge_stake": round(hedge_stake, 2)
                })

    return pd.DataFrame(recommendations)


# 3. Risk Reduction Hedging
def risk_reduction_hedging(initial_bets, current_odds_list, model_probs_list, risk_threshold=0.05):
    """
    Reduce the risk of a portfolio by allocating hedge bets.

    Parameters:
        initial_bets (list): List of dictionaries for initial bets.
        current_odds_list (list): Current odds for all outcomes.
        model_probs_list (list): Model probabilities for all outcomes.
        risk_threshold (float): Maximum acceptable risk threshold.

    Returns:
        pd.DataFrame: Summary of risk reduction hedging recommendations.
    """
    recommendations = []

    for i, bet in enumerate(initial_bets):
        outcome = bet["outcome"]
        odds = bet["odds"]
        stake = bet["stake"]

        current_odds = current_odds_list[i]
        model_probs = model_probs_list[i]

        for hedge_outcome, hedge_odds in current_odds.items():
            if hedge_outcome != outcome:
                hedge_prob = model_probs.get(hedge_outcome, 0)

                # Expected Return
                ev_initial = model_probs[outcome] * (odds - 1) - stake
                ev_hedge = hedge_prob * (hedge_odds - 1)

                if ev_initial - ev_hedge > risk_threshold:
                    hedge_stake = kelly_criterion(hedge_prob, hedge_odds) * stake
                    recommendations.append({
                        "initial_outcome": outcome,
                        "hedge_outcome": hedge_outcome,
                        "hedge_odds": hedge_odds,
                        "hedge_prob": hedge_prob,
                        "hedge_stake": round(hedge_stake, 2),
                        "risk_reduction": round(ev_initial - ev_hedge, 4)
                    })

    return pd.DataFrame(recommendations)


# 4. Arbitrage Opportunities
def identify_arbitrage_opportunities(games, total_stake=100):
    """
    Identify arbitrage opportunities and calculate optimal stake distribution.

    Parameters:
        games (list): List of games with odds for each outcome.
        total_stake (float): Total stake amount.

    Returns:
        pd.DataFrame: Summary of arbitrage opportunities.
    """
    opportunities = []

    for game in games:
        outcomes = game.get("outcomes", {})
        implied_probs = {k: 1 / v for k, v in outcomes.items()}
        implied_prob_sum = sum(implied_probs.values())

        if implied_prob_sum < 1:  # Arbitrage exists
            stakes = {k: (total_stake / v) / implied_prob_sum for k, v in outcomes.items()}
            guaranteed_profit = total_stake - sum(stakes.values())
            opportunities.append({
                "game": game["game"],
                "implied_prob_sum": implied_prob_sum,
                "stakes": stakes,
                "guaranteed_profit": round(guaranteed_profit, 2)
            })

    return pd.DataFrame(opportunities)


# 5. Probabilistic Arbitrage
def probabilistic_arbitrage_opportunities(games, model, total_stake=100):
    """
    Identify probabilistic arbitrage opportunities based on model probabilities.

    Parameters:
        games (list): List of games with odds for each outcome.
        model (sklearn.base.BaseEstimator): Trained model for predicting probabilities.
        total_stake (float): Total stake amount.

    Returns:
        pd.DataFrame: Summary of probabilistic arbitrage opportunities.
    """
    opportunities = []

    for game in games:
        outcomes = game.get("outcomes", {})
        X = np.array([[odds for _, odds in outcomes.items()]])
        model_probs = model.predict_proba(X)[0]  # Predict probabilities

        evs = {outcome: model_probs[i] * odds for i, (outcome, odds) in enumerate(outcomes.items())}
        positive_ev_outcomes = {k: v for k, v in evs.items() if v > 1}

        if positive_ev_outcomes:
            stakes = {k: (ev / sum(positive_ev_outcomes.values())) * total_stake for k, ev in positive_ev_outcomes.items()}
            opportunities.append({
                "game": game["game"],
                "positive_ev_outcomes": positive_ev_outcomes,
                "stakes": stakes
            })

    return pd.DataFrame(opportunities)

# 6. Hedge Bets
def hedge_bets(initial_bet, current_odds, hedge_outcome):
    """
    Calculate the optimal hedge bet to secure profit or minimize risk.

    Parameters:
        initial_bet (dict): Initial bet with keys 'stake', 'odds', and 'outcome'.
        current_odds (dict): Current odds for all outcomes.
        hedge_outcome (str): Outcome to hedge.

    Returns:
        dict: Hedge bet details.
    """
    stake = initial_bet["stake"]
    odds = initial_bet["odds"]
    hedge_odds = current_odds.get(hedge_outcome, 0)

    if hedge_odds > 0:
        hedge_stake = stake * odds / hedge_odds
        return {
            "hedge_outcome": hedge_outcome,
            "hedge_odds": hedge_odds,
            "hedge_stake": round(hedge_stake, 2)
        }
    return {"error": "Invalid hedge outcome or odds."}
