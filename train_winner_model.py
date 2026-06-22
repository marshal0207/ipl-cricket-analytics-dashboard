import os
import pickle
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


MATCH_FILE = "data/cleaned_matches_enriched.csv"
STRENGTH_FILE = "data/advanced_team_strength.csv"
PLAYER_STRENGTH_FILE = "data/match_player_strength.csv"

MODEL_FILE = "models/winner_model.pkl"
TEAM_ENCODER_FILE = "models/team_encoder.pkl"
WINNER_ENCODER_FILE = "models/winner_encoder.pkl"
VENUE_ENCODER_FILE = "models/venue_encoder.pkl"
TOSS_ENCODER_FILE = "models/toss_encoder.pkl"
FEATURES_FILE = "models/model_features.pkl"


def safe_win_pct(team, df):
    team_matches = df[
        (df["batting_team"] == team) |
        (df["bowling_team"] == team)
    ]

    if len(team_matches) == 0:
        return 0

    wins = len(team_matches[team_matches["match_won_by"] == team])
    return round((wins / len(team_matches)) * 100, 2)


def recent_form_score(team, df, before_index, last_n=10):
    previous_matches = df.iloc[:before_index]

    team_matches = previous_matches[
        (previous_matches["batting_team"] == team) |
        (previous_matches["bowling_team"] == team)
    ].tail(last_n)

    if len(team_matches) == 0:
        return 0

    wins = len(team_matches[team_matches["match_won_by"] == team])
    return round((wins / len(team_matches)) * 100, 2)


def h2h_wins(team, opponent, df, before_index):
    previous_matches = df.iloc[:before_index]

    h2h = previous_matches[
        (
            (previous_matches["batting_team"] == team) &
            (previous_matches["bowling_team"] == opponent)
        ) |
        (
            (previous_matches["batting_team"] == opponent) &
            (previous_matches["bowling_team"] == team)
        )
    ]

    return len(h2h[h2h["match_won_by"] == team])


def update_elo(winner_rating, loser_rating, k=32):
    expected_winner = 1 / (1 + 10 ** ((loser_rating - winner_rating) / 400))
    expected_loser = 1 / (1 + 10 ** ((winner_rating - loser_rating) / 400))

    new_winner_rating = winner_rating + k * (1 - expected_winner)
    new_loser_rating = loser_rating + k * (0 - expected_loser)

    return new_winner_rating, new_loser_rating

def load_strength_map():
    if not os.path.exists(STRENGTH_FILE):
        print("Warning: advanced_team_strength.csv not found.")
        return {}

    strength_df = pd.read_csv(STRENGTH_FILE)
    strength_map = {}

    for _, row in strength_df.iterrows():
        team = row["team"]
        strength_map[team] = {
            "batting": float(row.get("batting_rating", 0)),
            "bowling": float(row.get("bowling_rating", 0)),
            "strength": float(row.get("team_strength", 0)),
            "strike_rate": float(row.get("strike_rate", 0)),
            "economy": float(row.get("economy", 0)),
            "wicket_rate": float(row.get("wicket_rate", 0))
        }

    return strength_map


def load_match_player_strength():
    if not os.path.exists(PLAYER_STRENGTH_FILE):
        print("Warning: match_player_strength.csv not found.")
        return {}

    df = pd.read_csv(PLAYER_STRENGTH_FILE)
    strength_map = {}

    for _, row in df.iterrows():
        match_id = int(row["match_id"])
        team = row["team"]
        strength_map[(match_id, team)] = float(row["playing_xi_strength"])

    return strength_map


def main():
    os.makedirs("models", exist_ok=True)

    df = pd.read_csv(MATCH_FILE)

    strength_map = load_strength_map()
    player_strength_map = load_match_player_strength()

    required_cols = [
        "match_id", "batting_team", "bowling_team", "match_won_by",
        "season", "venue", "toss_winner", "toss_decision"
    ]

    for col in required_cols:
        if col not in df.columns:
            raise Exception(f"Missing required column: {col}")

    df = df.dropna(subset=["match_id", "batting_team", "bowling_team", "match_won_by"])

    df = df[df["match_won_by"].astype(str).str.lower() != "no result"]
    df = df[df["match_won_by"].astype(str).str.lower() != "unknown"]
    df = df[df["match_won_by"].astype(str).str.lower() != "nan"]

    df["venue"] = df["venue"].fillna("Unknown Venue")
    df["toss_winner"] = df["toss_winner"].fillna("Unknown")
    df["toss_decision"] = df["toss_decision"].fillna("Unknown")

    df["match_id"] = pd.to_numeric(df["match_id"], errors="coerce").fillna(0).astype(int)
    df["season"] = pd.to_numeric(df["season"], errors="coerce").fillna(0).astype(int)

    df = df.reset_index(drop=True)

    all_teams = sorted(
        set(df["batting_team"].dropna().unique())
        | set(df["bowling_team"].dropna().unique())
        | set(df["match_won_by"].dropna().unique())
        | set(df["toss_winner"].dropna().unique())
    )

    all_venues = sorted(df["venue"].dropna().astype(str).unique())
    all_toss_decisions = sorted(df["toss_decision"].dropna().astype(str).unique())

    team_encoder = LabelEncoder()
    team_encoder.fit(all_teams)

    winner_encoder = LabelEncoder()
    winner_encoder.fit(all_teams)

    venue_encoder = LabelEncoder()
    venue_encoder.fit(all_venues)

    toss_encoder = LabelEncoder()
    toss_encoder.fit(all_toss_decisions)

    rows = []

    elo_ratings = {team: 1500 for team in all_teams}

    for i, row in df.iterrows():
        match_id = int(row["match_id"])
        team1 = row["batting_team"]
        team2 = row["bowling_team"]
        winner = row["match_won_by"]
        venue = str(row["venue"])
        toss_winner = row["toss_winner"]
        toss_decision = str(row["toss_decision"])
        season = int(row["season"])
        team1_elo = elo_ratings.get(team1, 1500)
        team2_elo = elo_ratings.get(team2, 1500)
        elo_diff = team1_elo - team2_elo

        if team1 not in all_teams or team2 not in all_teams or winner not in all_teams:
            continue

        team1_encoded = team_encoder.transform([team1])[0]
        team2_encoded = team_encoder.transform([team2])[0]
        winner_encoded = winner_encoder.transform([winner])[0]

        venue_encoded = venue_encoder.transform([venue])[0]
        toss_decision_encoded = toss_encoder.transform([toss_decision])[0]
        toss_winner_encoded = team_encoder.transform([toss_winner])[0] if toss_winner in all_teams else -1

        team1_win_pct = safe_win_pct(team1, df.iloc[:i])
        team2_win_pct = safe_win_pct(team2, df.iloc[:i])

        team1_h2h = h2h_wins(team1, team2, df, i)
        team2_h2h = h2h_wins(team2, team1, df, i)

        team1_form = recent_form_score(team1, df, i)
        team2_form = recent_form_score(team2, df, i)

        team1_batting = strength_map.get(team1, {}).get("batting", 0)
        team2_batting = strength_map.get(team2, {}).get("batting", 0)

        team1_bowling = strength_map.get(team1, {}).get("bowling", 0)
        team2_bowling = strength_map.get(team2, {}).get("bowling", 0)

        team1_strength = strength_map.get(team1, {}).get("strength", 0)
        team2_strength = strength_map.get(team2, {}).get("strength", 0)

        team1_strike_rate = strength_map.get(team1, {}).get("strike_rate", 0)
        team2_strike_rate = strength_map.get(team2, {}).get("strike_rate", 0)

        team1_economy = strength_map.get(team1, {}).get("economy", 0)
        team2_economy = strength_map.get(team2, {}).get("economy", 0)

        team1_wicket_rate = strength_map.get(team1, {}).get("wicket_rate", 0)
        team2_wicket_rate = strength_map.get(team2, {}).get("wicket_rate", 0)

        team1_player_strength = player_strength_map.get((match_id, team1), 100)
        team2_player_strength = player_strength_map.get((match_id, team2), 100)

        rows.append({
            "team1_encoded": team1_encoded,
            "team2_encoded": team2_encoded,

            "team1_win_pct": team1_win_pct,
            "team2_win_pct": team2_win_pct,
            "team1_h2h": team1_h2h,
            "team2_h2h": team2_h2h,
            "team1_form": team1_form,
            "team2_form": team2_form,

            "win_pct_diff": team1_win_pct - team2_win_pct,
            "h2h_diff": team1_h2h - team2_h2h,
            "form_diff": team1_form - team2_form,

            "team1_batting": team1_batting,
            "team2_batting": team2_batting,
            "team1_bowling": team1_bowling,
            "team2_bowling": team2_bowling,
            "batting_diff": team1_batting - team2_batting,
            "bowling_diff": team1_bowling - team2_bowling,

            "team1_strength": team1_strength,
            "team2_strength": team2_strength,
            "strength_diff": team1_strength - team2_strength,

            "team1_player_strength": team1_player_strength,
            "team2_player_strength": team2_player_strength,
            "player_strength_diff": team1_player_strength - team2_player_strength,

            "team1_strike_rate": team1_strike_rate,
            "team2_strike_rate": team2_strike_rate,
            "strike_rate_diff": team1_strike_rate - team2_strike_rate,

            "team1_economy": team1_economy,
            "team2_economy": team2_economy,
            "economy_diff": team2_economy - team1_economy,

            "team1_wicket_rate": team1_wicket_rate,
            "team2_wicket_rate": team2_wicket_rate,
            "wicket_rate_diff": team1_wicket_rate - team2_wicket_rate,

            "venue_encoded": venue_encoded,
            "season": season,

            "toss_winner_encoded": toss_winner_encoded,
            "toss_decision_encoded": toss_decision_encoded,
            "toss_winner_is_team1": 1 if toss_winner == team1 else 0,
            "toss_winner_is_team2": 1 if toss_winner == team2 else 0,

            "team1_elo": team1_elo,
            "team2_elo": team2_elo,
            "elo_diff": elo_diff,

            "winner_encoded": winner_encoded
        })

        if winner == team1:
            new_team1_elo, new_team2_elo = update_elo(team1_elo, team2_elo)
        elif winner == team2:
            new_team2_elo, new_team1_elo = update_elo(team2_elo, team1_elo)
        else:
            new_team1_elo, new_team2_elo = team1_elo, team2_elo

        elo_ratings[team1] = new_team1_elo
        elo_ratings[team2] = new_team2_elo

    model_df = pd.DataFrame(rows)

    features = [col for col in model_df.columns if col != "winner_encoded"]

    X = model_df[features]
    y = model_df["winner_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.value_counts().min() > 1 else None
    )

    rf = RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    )

    param_grid = {
        "n_estimators": [200, 300, 500, 700],
        "max_depth": [6, 8, 10, 12],
        "min_samples_split": [5, 10, 15],
        "min_samples_leaf": [2, 4, 6],
        "max_features": ["sqrt", "log2"]
    }

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=30,
        cv=5,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
        verbose=2
    )

    search.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(search.best_params_)

    best_rf = search.best_estimator_

    model = CalibratedClassifierCV(
        estimator=best_rf,
        method="sigmoid",
        cv=5
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    with open(TEAM_ENCODER_FILE, "wb") as f:
        pickle.dump(team_encoder, f)

    with open(WINNER_ENCODER_FILE, "wb") as f:
        pickle.dump(winner_encoder, f)

    with open(VENUE_ENCODER_FILE, "wb") as f:
        pickle.dump(venue_encoder, f)

    with open(TOSS_ENCODER_FILE, "wb") as f:
        pickle.dump(toss_encoder, f)

    with open(FEATURES_FILE, "wb") as f:
        pickle.dump(features, f)

    print("\nModel trained successfully.")
    print(f"Accuracy: {round(accuracy * 100, 2)}%")

    print("\nFeatures used:")
    for feature in features:
        print("-", feature)


if __name__ == "__main__":
    main()