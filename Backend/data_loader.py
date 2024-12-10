import pandas as pd

def load_data(filepath):
    """Load NFL game data from CSV"""
    df = pd.read_csv(filepath)
    return df

# Example usage
def main():
    df = load_data('data/nfl/spreadspoke_scores.csv')