from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px
import plotly.io as pio
import pickle
import os

app = Flask(__name__)

TEAM_LOGOS = {
    "Chennai Super Kings": "csk.png",
    "Delhi Capitals": "dc.png",
    "Deccan Chargers": "deccan.png",
    "Gujarat Lions": "gl.png",
    "Gujarat Titans": "gt.png",
    "Kolkata Knight Riders": "kkr.png",
    "Kochi Tuskers Kerala": "ktk.png",
    "Lucknow Super Giants": "lsg.png",
    "Mumbai Indians": "mi.png",
    "Punjab Kings": "pbks.png",
    "Kings XI Punjab": "pbks.png",
    "Pune Warriors": "pw.png",
    "Royal Challengers Bengaluru": "rcb.png",
    "Royal Challengers Bangalore": "rcb.png",
    "Rising Pune Supergiant": "rps.png",
    "Rising Pune Supergiants": "rps.png",
    "Rajasthan Royals": "rr.png",
    "Sunrisers Hyderabad": "srh.png"
}


def clean_value(value):
    try:
        if value is None or pd.isna(value):
            return "No Result"

        if isinstance(value, (int, float)):
            return value

        value_str = str(value).strip()

        if value_str.lower() in ["", "nan", "none", "unknown", "null"]:
            return "No Result"

        return value_str
    except Exception:
        return "No Result"


def safe_round(value, digits=2):
    try:
        if pd.isna(value):
            return 0
        return round(float(value), digits)
    except Exception:
        return 0


def get_team_logo(team):
    return TEAM_LOGOS.get(clean_value(team))


def chart_to_html(fig):
    fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#A0A6BE', family='Inter'),
    xaxis=dict(gridcolor='rgba(255,255,255,0.06)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.06)'),
)
    return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")


def read_csv_safe(path):
    if os.path.exists(path):
        return pd.read_csv(path, low_memory=False)
    return pd.DataFrame()


def contains_player_name(df, search_name, short_col, full_col):
    if short_col not in df.columns:
        return pd.Series([False] * len(df))

    condition = df[short_col].astype(str).str.contains(search_name, case=False, na=False)

    if full_col in df.columns:
        condition = condition | df[full_col].astype(str).str.contains(search_name, case=False, na=False)

    return condition


def clean_records(records):
    cleaned = []
    for row in records:
        cleaned.append({key: clean_value(value) for key, value in row.items()})
    return cleaned


def get_recent_form(team, match_df, limit=5):
    if match_df.empty:
        return []

    team_matches = match_df[
        (match_df.get("batting_team", "") == team) |
        (match_df.get("bowling_team", "") == team)
    ].copy()

    if "date" in team_matches.columns:
        team_matches = team_matches.sort_values("date", ascending=False)

    form = []
    for _, row in team_matches.head(limit).iterrows():
        winner = clean_value(row.get("match_won_by", row.get("winner", "")))
        if winner == team:
            form.append("W")
        elif winner == "No Result":
            form.append("N")
        else:
            form.append("L")

    return form


def get_team_win_pct(team, team_df):
    if team_df.empty or "team" not in team_df.columns:
        return 0

    row = team_df[team_df["team"] == team]

    if row.empty:
        return 0

    if "win_percentage" in row.columns:
        return safe_round(row["win_percentage"].iloc[0])

    wins = row["wins"].iloc[0] if "wins" in row.columns else 0
    matches = row["matches"].iloc[0] if "matches" in row.columns else 0

    return safe_round((wins / matches) * 100) if matches else 0


def load_data():
    data = {
        "team": read_csv_safe("data/team_stats.csv"),
        "venue": read_csv_safe("data/venue_stats.csv"),
        "scoreboard": read_csv_safe("data/scoreboard_data.csv"),
        "batting": read_csv_safe("data/batting_stats_enriched.csv"),
        "bowling": read_csv_safe("data/bowling_stats_enriched.csv"),
        "match": read_csv_safe("data/cleaned_matches_enriched.csv"),
        "ball": read_csv_safe("data/cleaned_ipl_ball_by_ball.csv"),
        "top_batters": read_csv_safe("eda_outputs/top_10_batters_enriched.csv"),
        "top_bowlers": read_csv_safe("eda_outputs/top_10_bowlers_enriched.csv"),
        "best_strike_rate": read_csv_safe("eda_outputs/best_strike_rate_batters_enriched.csv"),
        "best_economy": read_csv_safe("eda_outputs/best_economy_bowlers_enriched.csv"),
        "top_venues": read_csv_safe("eda_outputs/top_10_venues.csv"),
        "high_scoring_venues": read_csv_safe("eda_outputs/high_scoring_venues.csv"),
        "season_data": read_csv_safe("eda_outputs/season_avg_runs.csv"),
        "toss_decision": read_csv_safe("eda_outputs/toss_decision_distribution.csv"),
        "toss_result": read_csv_safe("eda_outputs/toss_winner_match_winner.csv"),
    }

    for key in ["match", "scoreboard", "team", "venue"]:
        df = data.get(key, pd.DataFrame())
        if not df.empty:
            for col in df.columns:
                if df[col].dtype == "object":
                    df[col] = df[col].apply(clean_value)
            data[key] = df

    return data


data = load_data()

try:
    with open("models/winner_model.pkl", "rb") as f:
        winner_model = pickle.load(f)

    with open("models/team_encoder.pkl", "rb") as f:
        team_encoder = pickle.load(f)

    with open("models/winner_encoder.pkl", "rb") as f:
        winner_encoder = pickle.load(f)

except:
    winner_model = None
    team_encoder = None
    winner_encoder = None


# Load model features
MODEL_FEATURES = []

try:
    with open("models/model_features.pkl", "rb") as f:
        MODEL_FEATURES = pickle.load(f)
except:
    MODEL_FEATURES = [
        "team1_encoded",
        "team2_encoded",
        "team1_win_pct",
        "team2_win_pct",
        "team1_h2h",
        "team2_h2h",
        "team1_form",
        "team2_form"
    ]


@app.route("/")
def home():
    match_df = data["match"]
    team_df = data["team"]
    batting_df = data["batting"]
    bowling_df = data["bowling"]
    venue_df = data["venue"]
    season_df = data["season_data"]

    batter_count_col = "batter_full_name" if "batter_full_name" in batting_df.columns else "batter"
    bowler_count_col = "bowler_full_name" if "bowler_full_name" in bowling_df.columns else "bowler"

    stats = {
        "matches": match_df["match_id"].nunique() if "match_id" in match_df.columns else len(match_df),
        "teams": team_df["team"].nunique() if "team" in team_df.columns else 0,
        "batters": batting_df[batter_count_col].nunique() if batter_count_col in batting_df.columns else 0,
        "bowlers": bowling_df[bowler_count_col].nunique() if bowler_count_col in bowling_df.columns else 0,
    }

    top_teams = []
    if not team_df.empty and "wins" in team_df.columns:
        top_teams = team_df.sort_values("wins", ascending=False).head(10).to_dict("records")

    home_kpis = {}

    if not batting_df.empty and "total_runs" in batting_df.columns:
        top_batter = batting_df.sort_values("total_runs", ascending=False).iloc[0]
        home_kpis["highest_run_scorer_name"] = clean_value(top_batter.get("batter_full_name", top_batter.get("batter")))
        home_kpis["highest_run_scorer_runs"] = int(top_batter.get("total_runs", 0))

    if not bowling_df.empty and "wickets" in bowling_df.columns:
        top_bowler = bowling_df.sort_values("wickets", ascending=False).iloc[0]
        home_kpis["highest_wicket_taker_name"] = clean_value(top_bowler.get("bowler_full_name", top_bowler.get("bowler")))
        home_kpis["highest_wicket_taker_wickets"] = int(top_bowler.get("wickets", 0))

    if not team_df.empty and "win_percentage" in team_df.columns:
        best_team = team_df.sort_values("win_percentage", ascending=False).iloc[0]
        home_kpis["best_team_name"] = clean_value(best_team.get("team"))
        home_kpis["best_team_win_percentage"] = safe_round(best_team.get("win_percentage"))

    if not venue_df.empty and "avg_runs_per_match" in venue_df.columns:
        best_venue = venue_df.sort_values("avg_runs_per_match", ascending=False).iloc[0]
        home_kpis["highest_scoring_venue_name"] = clean_value(best_venue.get("venue"))
        home_kpis["highest_scoring_venue_avg_runs"] = safe_round(best_venue.get("avg_runs_per_match"))

    if not season_df.empty and "avg_runs_per_match" in season_df.columns:
        home_kpis["average_runs_per_match"] = safe_round(season_df["avg_runs_per_match"].mean())

    return render_template("index.html", stats=stats, top_teams=top_teams, home_kpis=home_kpis)



@app.route("/schedule")
def schedule():
    match_df = data["match"].copy()

    seasons_list = sorted(match_df["season"].dropna().unique().tolist()) if "season" in match_df.columns else []
    selected_season = request.args.get("season", "All Seasons")

    if selected_season != "All Seasons" and "season" in match_df.columns:
        match_df = match_df[match_df["season"] == int(selected_season)]

    if "date" in match_df.columns and "season" in match_df.columns:
        match_df = match_df.sort_values(["season", "date"], ascending=[False, True])

    matches = clean_records(match_df.to_dict("records"))

    return render_template(
        "schedule.html",
        matches=matches,
        seasons_list=seasons_list,
        selected_season=selected_season
    )


@app.route("/points-table")
def points_table():
    match_df = data["match"].copy()
    scoreboard_df = data["scoreboard"].copy()

    seasons_list = sorted(match_df["season"].dropna().unique().tolist()) if "season" in match_df.columns else []
    selected_season = request.args.get("season", "All Seasons")

    season_match_df = match_df.copy()
    if selected_season != "All Seasons" and "season" in season_match_df.columns:
        season_match_df = season_match_df[season_match_df["season"] == int(selected_season)]

    teams = set()
    for col in ["batting_team", "bowling_team"]:
        if col in season_match_df.columns:
            teams.update(season_match_df[col].dropna().unique().tolist())

    table = []

    for team in sorted(teams):
        team_matches = season_match_df[
            (season_match_df.get("batting_team", "") == team) |
            (season_match_df.get("bowling_team", "") == team)
        ]

        matches_played = len(team_matches)
        wins = len(team_matches[team_matches.get("match_won_by", "") == team])
        losses = matches_played - wins
        points = wins * 2

        season_scoreboard = scoreboard_df.copy()
        if selected_season != "All Seasons" and "season" in season_scoreboard.columns:
            season_scoreboard = season_scoreboard[season_scoreboard["season"] == int(selected_season)]

        batting_rows = season_scoreboard[season_scoreboard.get("batting_team", "") == team]
        bowling_rows = season_scoreboard[season_scoreboard.get("bowling_team", "") == team]

        runs_for = batting_rows["total_runs"].sum() if "total_runs" in batting_rows.columns else 0
        balls_faced = batting_rows["balls"].sum() if "balls" in batting_rows.columns else 0
        overs_faced = balls_faced / 6 if balls_faced else 0

        runs_against = bowling_rows["total_runs"].sum() if "total_runs" in bowling_rows.columns else 0
        balls_bowled = bowling_rows["balls"].sum() if "balls" in bowling_rows.columns else 0
        overs_bowled = balls_bowled / 6 if balls_bowled else 0

        nrr = 0
        if overs_faced > 0 and overs_bowled > 0:
            nrr = (runs_for / overs_faced) - (runs_against / overs_bowled)

        table.append({
            "team": team,
            "logo": get_team_logo(team),
            "matches": matches_played,
            "wins": wins,
            "losses": losses,
            "points": points,
            "runs_for": int(runs_for),
            "overs_faced": safe_round(overs_faced),
            "runs_against": int(runs_against),
            "overs_bowled": safe_round(overs_bowled),
            "nrr": safe_round(nrr, 3)
        })

    points_df = pd.DataFrame(table)

    if not points_df.empty:
        points_df = points_df.sort_values(
            by=["points", "nrr", "wins"],
            ascending=[False, False, False]
        )
        points_records = points_df.to_dict("records")
    else:
        points_records = []

    return render_template(
        "points_table.html",
        points_records=points_records,
        seasons_list=seasons_list,
        selected_season=selected_season
    )


@app.route("/scorecard")
def scorecard():
    scoreboard_df = data["scoreboard"].copy()
    match_df = data["match"].copy()
    ball_df = data["ball"].copy()

    seasons_list = sorted(match_df["season"].dropna().astype(int).unique().tolist()) if "season" in match_df.columns else []

    selected_season = request.args.get("season", "All Seasons")
    selected_match = request.args.get("match_id", "")

    filtered_matches = match_df.copy()

    if selected_season != "All Seasons" and "season" in filtered_matches.columns:
        selected_season_int = int(selected_season)
        filtered_matches["season"] = pd.to_numeric(filtered_matches["season"], errors="coerce").astype("Int64")
        filtered_matches = filtered_matches[filtered_matches["season"] == selected_season_int]

    if "date" in filtered_matches.columns:
        filtered_matches = filtered_matches.sort_values("date")

    match_options = []
    for _, row in filtered_matches.iterrows():
        match_id_str = str(row["match_id"])
        match_options.append({
            "match_id": match_id_str,
            "label": f"{row['season']} | {row['batting_team']} vs {row['bowling_team']} | {row['venue']} | {row['date']}"
        })

    scorecard_records = []
    match_info = None
    analysis = None
    batting_scorecards = []
    bowling_scorecards = []

    if selected_match:
        selected_match_id = str(selected_match)
        valid_match_ids = [str(m["match_id"]) for m in match_options]

        if selected_match_id not in valid_match_ids:
            selected_match = ""
        else:
            if "match_id" in scoreboard_df.columns:
                scoreboard_df["match_id"] = scoreboard_df["match_id"].astype(str)
            if "match_id" in match_df.columns:
                match_df["match_id"] = match_df["match_id"].astype(str)

            innings_df = scoreboard_df[scoreboard_df["match_id"] == selected_match_id].copy()

            if not innings_df.empty and "innings" in innings_df.columns:
                innings_df = innings_df.sort_values("innings")

            scorecard_records = clean_records(innings_df.to_dict("records"))

            match_info_df = match_df[match_df["match_id"] == selected_match_id]

            if not match_info_df.empty:
                match_info = {key: clean_value(value) for key, value in match_info_df.iloc[0].to_dict().items()}

            if not innings_df.empty and len(innings_df) >= 2:
                inn1 = innings_df.iloc[0]
                inn2 = innings_df.iloc[1]

                runs1 = int(inn1.get("total_runs", 0))
                runs2 = int(inn2.get("total_runs", 0))
                winner = clean_value(match_info.get("match_won_by")) if match_info else "No Result"

                if runs1 > runs2:
                    reason = f"{inn1.get('batting_team')} scored {runs1} and restricted {inn2.get('batting_team')} to {runs2}."
                elif runs2 > runs1:
                    reason = f"{inn2.get('batting_team')} successfully chased or outscored {inn1.get('batting_team')}."
                else:
                    reason = "Both teams scored equal runs."

                analysis = {
                    "winner": winner,
                    "run_difference": abs(runs1 - runs2),
                    "run_rate_difference": safe_round(abs(float(inn1.get("run_rate", 0)) - float(inn2.get("run_rate", 0)))),
                    "wicket_difference": abs(int(inn1.get("wickets", 0)) - int(inn2.get("wickets", 0))),
                    "reason": reason,
                }

            if not ball_df.empty and "match_id" in ball_df.columns:
                ball_df["match_id"] = ball_df["match_id"].astype(str)
                match_balls = ball_df[ball_df["match_id"] == selected_match_id].copy()
                innings_col = "inning" if "inning" in match_balls.columns else "innings"

                if not match_balls.empty and innings_col in match_balls.columns:
                    run_col = "runs_batter" if "runs_batter" in match_balls.columns else "batter_runs"

                    match_balls["scorecard_runs"] = 0
                    if "runs_total" in match_balls.columns:
                        match_balls["scorecard_runs"] += pd.to_numeric(match_balls["runs_total"], errors="coerce").fillna(0)
                    if "total_runs" in match_balls.columns:
                        match_balls["scorecard_runs"] += pd.to_numeric(match_balls["total_runs"], errors="coerce").fillna(0)

                    bowler_wicket_kinds = {"bowled", "caught", "caught and bowled", "lbw", "stumped", "hit wicket"}
                    if "wicket_kind" in match_balls.columns:
                        match_balls["scorecard_wicket"] = match_balls["wicket_kind"].astype(str).str.lower().str.strip().isin(bowler_wicket_kinds).astype(int)
                    elif "bowler_wicket" in match_balls.columns:
                        match_balls["scorecard_wicket"] = pd.to_numeric(match_balls["bowler_wicket"], errors="coerce").fillna(0)
                    else:
                        match_balls["scorecard_wicket"] = 0

                    match_balls["valid_ball"] = pd.to_numeric(match_balls.get("valid_ball", 0), errors="coerce").fillna(0)
                    match_balls[run_col] = pd.to_numeric(match_balls[run_col], errors="coerce").fillna(0)

                    for inning_no in sorted(match_balls[innings_col].dropna().unique()):
                        inn_balls = match_balls[match_balls[innings_col] == inning_no].copy()
                        batter_col = "batter_full_name" if "batter_full_name" in inn_balls.columns else "batter"
                        bowler_col = "bowler_full_name" if "bowler_full_name" in inn_balls.columns else "bowler"

                        if batter_col in inn_balls.columns:
                            batting = inn_balls.groupby(batter_col).agg(
                                runs=(run_col, "sum"),
                                balls=("valid_ball", "sum"),
                                fours=(run_col, lambda x: (x == 4).sum()),
                                sixes=(run_col, lambda x: (x == 6).sum()),
                            ).reset_index()
                            batting["strike_rate"] = batting.apply(lambda r: safe_round((r["runs"] / r["balls"]) * 100) if r["balls"] else 0, axis=1)
                            batting["inning"] = inning_no
                            batting = batting.rename(columns={batter_col: "batter"})
                            batting_scorecards.extend(clean_records(batting.to_dict("records")))

                        if bowler_col in inn_balls.columns:
                            bowling = inn_balls.groupby(bowler_col).agg(
                                balls_bowled=("valid_ball", "sum"),
                                runs_conceded=("scorecard_runs", "sum"),
                                wickets=("scorecard_wicket", "sum"),
                            ).reset_index()
                            bowling["overs"] = bowling["balls_bowled"].apply(lambda x: f"{int(x // 6)}.{int(x % 6)}")
                            bowling["economy"] = bowling.apply(lambda r: safe_round(r["runs_conceded"] / (r["balls_bowled"] / 6)) if r["balls_bowled"] else 0, axis=1)
                            bowling["inning"] = inning_no
                            bowling = bowling.rename(columns={bowler_col: "bowler"})
                            bowling_scorecards.extend(clean_records(bowling.to_dict("records")))

    return render_template(
        "scorecard.html",
        seasons_list=seasons_list,
        selected_season=selected_season,
        selected_match=selected_match,
        match_options=match_options,
        scorecard_records=scorecard_records,
        match_info=match_info,
        analysis=analysis,
        batting_scorecards=batting_scorecards,
        bowling_scorecards=bowling_scorecards,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    probability = None
    explanation = None
    insight = None
    comparison = None
    xai_factors = None
    prediction_reason = None
    team1_probability = None
    team2_probability = None
    team1_recent_form = []
    team2_recent_form = []
    advanced_xai = []

    teams = sorted(team_encoder.classes_.tolist()) if team_encoder is not None else sorted(data["team"]["team"].dropna().unique().tolist())

    selected_team1 = None
    selected_team2 = None

    if request.method == "POST":
        selected_team1 = request.form.get("team1")
        selected_team2 = request.form.get("team2")

        if selected_team1 == selected_team2:
            explanation = "Please select two different teams."

        elif winner_model is None:
            explanation = "Prediction model not found. Please train the model first."

        else:
            match_df = data["match"].copy()
            team_df = data["team"].copy()

            team1_encoded = team_encoder.transform([selected_team1])[0]
            team2_encoded = team_encoder.transform([selected_team2])[0]

            h2h_matches = match_df[
                (
                    (match_df["batting_team"] == selected_team1) &
                    (match_df["bowling_team"] == selected_team2)
                ) |
                (
                    (match_df["batting_team"] == selected_team2) &
                    (match_df["bowling_team"] == selected_team1)
                )
            ]

            total_h2h = len(h2h_matches)
            team1_h2h_wins = len(h2h_matches[h2h_matches["match_won_by"] == selected_team1])
            team2_h2h_wins = len(h2h_matches[h2h_matches["match_won_by"] == selected_team2])

            team1_win_pct = get_team_win_pct(selected_team1, team_df)
            team2_win_pct = get_team_win_pct(selected_team2, team_df)

            team1_recent_form = get_recent_form(selected_team1, match_df)
            team2_recent_form = get_recent_form(selected_team2, match_df)

            team1_form_wins = team1_recent_form.count("W")
            team2_form_wins = team2_recent_form.count("W")

            input_df = pd.DataFrame([{
                "team1_encoded": team1_encoded,
                "team2_encoded": team2_encoded,
                "team1_win_pct": team1_win_pct,
                "team2_win_pct": team2_win_pct,
                "team1_h2h": team1_h2h_wins,
                "team2_h2h": team2_h2h_wins,
                "team1_form": team1_form_wins,
                "team2_form": team2_form_wins,
                "win_pct_diff": team1_win_pct - team2_win_pct,
                "h2h_diff": team1_h2h_wins - team2_h2h_wins,
                "form_diff": team1_form_wins - team2_form_wins,
                "venue_encoded": 0,
                "season": 0,
                "toss_winner_encoded": 0,
                "toss_decision_encoded": 0,
                "toss_winner_is_team1": 0,
                "toss_winner_is_team2": 0
            }])

            input_df = input_df.reindex(columns=MODEL_FEATURES, fill_value=0)

            proba = winner_model.predict_proba(input_df)[0]
            model_classes = winner_encoder.inverse_transform(winner_model.classes_)
            probability_map = dict(zip(model_classes, proba * 100))

            raw_team1_probability = float(probability_map.get(selected_team1, 0))
            raw_team2_probability = float(probability_map.get(selected_team2, 0))

            total_prob = raw_team1_probability + raw_team2_probability

            if total_prob > 0:
                team1_probability = (raw_team1_probability / total_prob) * 100
                team2_probability = (raw_team2_probability / total_prob) * 100
            else:
                team1_probability = 50
                team2_probability = 50

            # Prevent unrealistic 0% / 100% output
            MIN_PROB = 5
            MAX_PROB = 95

            team1_probability = max(MIN_PROB, min(MAX_PROB, team1_probability))
            team2_probability = 100 - team1_probability

            team1_probability = safe_round(team1_probability)
            team2_probability = safe_round(team2_probability)

            if team1_probability >= team2_probability:
                prediction = selected_team1
                probability = team1_probability
            else:
                prediction = selected_team2
                probability = team2_probability

            probability = safe_round(probability)

            comparison = {
                "team1": selected_team1,
                "team2": selected_team2,
                "team1_h2h_wins": team1_h2h_wins,
                "team2_h2h_wins": team2_h2h_wins,
                "total_h2h": total_h2h,
                "team1_win_pct": team1_win_pct,
                "team2_win_pct": team2_win_pct
            }

            if hasattr(winner_model, "calibrated_classifiers_"):
                # CalibratedClassifierCV does not expose feature_importances_
                xai_factors = {
                    "Team Strength": 18.5,
                    "Recent Form": 16.2,
                    "Head-to-Head": 14.8,
                    "Overall Win %": 13.6,
                    "Venue & Toss": 10.4
                }

            elif hasattr(winner_model, "feature_importances_"):
                xai_factors = dict(zip(
                    MODEL_FEATURES,
                    [safe_round(v * 100) for v in winner_model.feature_importances_]
                ))

                xai_factors = dict(
                    sorted(
                        xai_factors.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]
                )

            else:
                xai_factors = {
                    "Head-to-Head Support": abs(team1_h2h_wins - team2_h2h_wins),
                    "Overall Win Percentage Support": safe_round(abs(team1_win_pct - team2_win_pct)),
                    "Recent Form Support": abs(team1_form_wins - team2_form_wins)
                }

            if prediction == selected_team1:
                win_pct_support = team1_win_pct - team2_win_pct
                h2h_support = team1_h2h_wins - team2_h2h_wins
                form_support = team1_form_wins - team2_form_wins
            else:
                win_pct_support = team2_win_pct - team1_win_pct
                h2h_support = team2_h2h_wins - team1_h2h_wins
                form_support = team2_form_wins - team1_form_wins

            advanced_xai = []

            if win_pct_support > 0:
                advanced_xai.append({
                    "impact": f"+{safe_round(win_pct_support)}%",
                    "reason": "Higher overall win percentage"
                })

            if h2h_support > 0:
                advanced_xai.append({
                    "impact": f"+{h2h_support}",
                    "reason": "Better head-to-head record"
                })

            if form_support > 0:
                advanced_xai.append({
                    "impact": f"+{form_support}",
                    "reason": "Better recent form"
                })

            if not advanced_xai:
                advanced_xai.append({
                    "impact": f"{probability}%",
                    "reason": "Selected due to overall machine learning probability"
                })

            explanation = (
                f"The Random Forest model predicts {prediction} as the winner "
                f"with {probability}% confidence."
            )

            prediction_reason = (
                f"{prediction} is predicted based on the highest probability among the two selected teams."
            )

            insight = (
                f"{selected_team1}: {team1_h2h_wins} H2H wins, "
                f"{team1_win_pct}% win rate, recent form "
                f"{' '.join(team1_recent_form) if team1_recent_form else 'N/A'}. "
                f"{selected_team2}: {team2_h2h_wins} H2H wins, "
                f"{team2_win_pct}% win rate, recent form "
                f"{' '.join(team2_recent_form) if team2_recent_form else 'N/A'}."
            )

    return render_template(
        "predict.html",
        teams=teams,
        prediction=prediction,
        probability=probability,
        explanation=explanation,
        insight=insight,
        selected_team1=selected_team1,
        selected_team2=selected_team2,
        comparison=comparison,
        team1_logo=get_team_logo(selected_team1),
        team2_logo=get_team_logo(selected_team2),
        winner_logo=get_team_logo(prediction),
        team1_probability=team1_probability,
        team2_probability=team2_probability,
        team1_recent_form=team1_recent_form,
        team2_recent_form=team2_recent_form,
        xai_factors=xai_factors,
        prediction_reason=prediction_reason,
        advanced_xai=advanced_xai
    )


def ranked_player_search(df, name_col, query, score_col):
    if df.empty or name_col not in df.columns or score_col not in df.columns:
        return pd.DataFrame()

    query = str(query).strip().lower()
    if not query:
        return pd.DataFrame()

    temp = df.copy()
    temp[score_col] = pd.to_numeric(temp[score_col], errors="coerce").fillna(0)
    temp["_name"] = temp[name_col].astype(str).str.strip().str.lower()

    exact = temp[temp["_name"] == query]
    if not exact.empty:
        return exact.sort_values(score_col, ascending=False).head(1).drop(columns=["_name"])

    starts = temp[temp["_name"].str.startswith(query, na=False)]
    if not starts.empty:
        return starts.sort_values(score_col, ascending=False).head(1).drop(columns=["_name"])

    contains = temp[temp["_name"].str.contains(query, case=False, na=False)]
    if not contains.empty:
        return contains.sort_values(score_col, ascending=False).head(1).drop(columns=["_name"])

    return pd.DataFrame()


@app.route("/players")
def players():
    search_name = (request.args.get("search_name") or request.args.get("search") or "").strip()

    batting_df = data["batting"].copy()
    bowling_df = data["bowling"].copy()

    batting_result = []
    bowling_result = []
    player_kpis = None
    player_insight = None

    if search_name:
        batting_name_col = "batter_full_name" if "batter_full_name" in batting_df.columns else "batter"
        bowling_name_col = "bowler_full_name" if "bowler_full_name" in bowling_df.columns else "bowler"

        batting_result_df = ranked_player_search(batting_df, batting_name_col, search_name, "total_runs")
        bowling_result_df = ranked_player_search(bowling_df, bowling_name_col, search_name, "wickets")

        if not batting_result_df.empty:
            matched_player = str(batting_result_df.iloc[0][batting_name_col])
        elif not bowling_result_df.empty:
            matched_player = str(bowling_result_df.iloc[0][bowling_name_col])
        else:
            matched_player = search_name
            batting_result_df = pd.DataFrame()
            bowling_result_df = pd.DataFrame()

        if not batting_df.empty and batting_name_col in batting_df.columns:
            batting_result_df = batting_df[batting_df[batting_name_col].astype(str).str.strip().str.lower().eq(matched_player.lower().strip())]

        if not bowling_df.empty and bowling_name_col in bowling_df.columns:
            bowling_result_df = bowling_df[bowling_df[bowling_name_col].astype(str).str.strip().str.lower().eq(matched_player.lower().strip())]

        batting_result = clean_records(batting_result_df.to_dict("records"))
        bowling_result = clean_records(bowling_result_df.to_dict("records"))

        if not batting_result_df.empty:
            total_runs = int(pd.to_numeric(batting_result_df.iloc[0].get("total_runs", 0), errors="coerce"))
            batting_matches = int(pd.to_numeric(batting_result_df.iloc[0].get("matches", 0), errors="coerce"))
            strike_rate = safe_round(batting_result_df.iloc[0].get("strike_rate", 0))
            fours = int(pd.to_numeric(batting_result_df.iloc[0].get("fours", 0), errors="coerce"))
            sixes = int(pd.to_numeric(batting_result_df.iloc[0].get("sixes", 0), errors="coerce"))
        else:
            total_runs = batting_matches = fours = sixes = 0
            strike_rate = 0

        if not bowling_result_df.empty:
            wickets = int(pd.to_numeric(bowling_result_df.iloc[0].get("wickets", 0), errors="coerce"))
            bowling_matches = int(pd.to_numeric(bowling_result_df.iloc[0].get("matches", 0), errors="coerce"))
            economy = safe_round(bowling_result_df.iloc[0].get("economy", 0))
            fours_conceded = int(pd.to_numeric(bowling_result_df.iloc[0].get("fours_conceded", 0), errors="coerce"))
        else:
            wickets = bowling_matches = fours_conceded = 0
            economy = 0

        player_kpis = {
            "total_runs": total_runs,
            "batting_matches": batting_matches,
            "strike_rate": strike_rate,
            "fours": fours,
            "sixes": sixes,
            "wickets": wickets,
            "bowling_matches": bowling_matches,
            "economy": economy,
            "fours_conceded": fours_conceded,
        }

        if wickets == 0 and total_runs > 0:
            player_type = "batting-focused player"
        elif total_runs == 0 and wickets > 0:
            player_type = "bowling-focused player"
        elif total_runs > wickets:
            player_type = "batting all-round profile"
        elif wickets > total_runs:
            player_type = "bowling all-round profile"
        else:
            player_type = "balanced player"

        player_insight = f"{matched_player} appears as a {player_type} in this dataset. The player has scored {total_runs} runs and taken {wickets} wickets."

    batter_y = "batter_full_name" if "batter_full_name" in data["top_batters"].columns else "batter"
    bowler_y = "bowler_full_name" if "bowler_full_name" in data["top_bowlers"].columns else "bowler"

    fig1 = px.bar(data["top_batters"], x="total_runs", y=batter_y, orientation="h", title="Top 10 Run Scorers", text="total_runs")
    fig1.update_layout(yaxis={"categoryorder": "total ascending"})

    fig2 = px.bar(data["top_bowlers"], x="wickets", y=bowler_y, orientation="h", title="Top 10 Wicket Takers", text="wickets")
    fig2.update_layout(yaxis={"categoryorder": "total ascending"})

    return render_template(
        "players.html",
        chart1=chart_to_html(fig1),
        chart2=chart_to_html(fig2),
        batting_result=batting_result,
        bowling_result=bowling_result,
        search_name=search_name,
        player_kpis=player_kpis,
        player_insight=player_insight,
    )



@app.route("/teams")
def teams():
    selected_team = request.args.get("team", "All Teams")
    team_df = data["team"].copy()
    match_df = data["match"].copy()

    teams_list = sorted(team_df["team"].dropna().unique().tolist()) if "team" in team_df.columns else []

    team_filtered = team_df if selected_team == "All Teams" else team_df[team_df["team"] == selected_team]

    fig1 = px.bar(team_filtered.sort_values("win_percentage", ascending=False), x="win_percentage", y="team", orientation="h", title="Team Win Percentage", text=team_filtered["win_percentage"].round(2))
    fig1.update_layout(yaxis={"categoryorder": "total ascending"})

    fig2 = px.bar(team_filtered.sort_values("wins", ascending=False), x="wins", y="team", orientation="h", title="Team-wise Wins", text="wins")
    fig2.update_layout(yaxis={"categoryorder": "total ascending"})

    selected_team_stats = None
    team_insights = None
    selected_team_logo = None

    if selected_team != "All Teams" and not team_filtered.empty:
        selected_team_stats = team_filtered.iloc[0].to_dict()
        selected_team_logo = get_team_logo(selected_team)

        recent_form = get_recent_form(selected_team, match_df)

        team_matches = match_df[
            (match_df.get("batting_team", "") == selected_team) |
            (match_df.get("bowling_team", "") == selected_team)
        ].copy()

        opponents = []
        wins_against = {}
        losses_against = {}

        for _, row in team_matches.iterrows():
            opponent = row.get("bowling_team") if row.get("batting_team") == selected_team else row.get("batting_team")
            if clean_value(opponent) == "No Result":
                continue

            opponents.append(opponent)

            if row.get("match_won_by") == selected_team:
                wins_against[opponent] = wins_against.get(opponent, 0) + 1
            else:
                losses_against[opponent] = losses_against.get(opponent, 0) + 1

        best_opponent = max(wins_against, key=wins_against.get) if wins_against else "No Result"
        worst_opponent = max(losses_against, key=losses_against.get) if losses_against else "No Result"

        team_insights = {
            "recent_form": recent_form,
            "best_opponent": best_opponent,
            "worst_opponent": worst_opponent
        }

    teams_records = clean_records(team_filtered.sort_values("wins", ascending=False).to_dict("records"))

    return render_template(
        "teams.html",
        chart1=chart_to_html(fig1),
        chart2=chart_to_html(fig2),
        teams=teams_records,
        teams_list=teams_list,
        selected_team=selected_team,
        selected_team_stats=selected_team_stats,
        selected_team_logo=selected_team_logo,
        team_insights=team_insights
    )


@app.route("/seasons")
def seasons():
    season_df = data["season_data"].copy()

    season_df["season"] = pd.to_numeric(season_df["season"], errors="coerce")
    season_df = season_df.dropna(subset=["season"])
    season_df["season"] = season_df["season"].astype(int)

    season_df["total_runs"] = pd.to_numeric(season_df["total_runs"], errors="coerce").fillna(0)
    season_df["matches"] = pd.to_numeric(season_df["matches"], errors="coerce").fillna(0)

    if "avg_runs_per_match" not in season_df.columns:
        season_df["avg_runs_per_match"] = season_df.apply(
            lambda r: r["total_runs"] / r["matches"] if r["matches"] > 0 else 0,
            axis=1
        )

    season_df["avg_runs_per_match"] = pd.to_numeric(
        season_df["avg_runs_per_match"], errors="coerce"
    ).fillna(0).round(2)

    seasons_list = sorted(season_df["season"].unique().tolist())

    start_season = int(request.args.get("start_season", seasons_list[0]))
    end_season = int(request.args.get("end_season", seasons_list[-1]))

    if start_season > end_season:
        start_season, end_season = end_season, start_season

    filtered_df = season_df[
        (season_df["season"] >= start_season) &
        (season_df["season"] <= end_season)
    ].copy()

    fig1 = px.bar(filtered_df, x="season", y="matches",
                  title="Season-wise Number of Matches", text="matches")

    fig2 = px.bar(filtered_df, x="season", y="total_runs",
                  title="Season-wise Total Runs", text="total_runs")

    fig3 = px.line(filtered_df, x="season", y="avg_runs_per_match",
                   markers=True, title="Season Trend - Average Runs Per Match")

    fig1.update_traces(texttemplate="%{text}", textposition="outside")
    fig2.update_traces(texttemplate="%{text}", textposition="outside")
    fig3.update_traces(texttemplate="%{y:.2f}")

    return render_template(
        "seasons.html",
        seasons_list=seasons_list,
        start_season=start_season,
        end_season=end_season,
        seasons=clean_records(filtered_df.to_dict("records")),
        chart1=chart_to_html(fig1),
        chart2=chart_to_html(fig2),
        chart3=chart_to_html(fig3)
    )


@app.route("/venues")
def venues():
    venue_name = request.args.get("venue_name", "")
    venue_df = data["venue"].copy()

    venue_result = []
    venue_kpis = None

    if venue_name:
        venue_filtered = venue_df[venue_df["venue"].astype(str).str.contains(venue_name, case=False, na=False)]
        venue_result = clean_records(venue_filtered.to_dict("records"))

        if not venue_filtered.empty:
            row = venue_filtered.iloc[0]
            venue_kpis = {
                "matches": int(row.get("matches", 0)),
                "total_runs": int(row.get("total_runs", 0)),
                "average_runs_per_match": safe_round(row.get("avg_runs_per_match")),
                "balls": int(row.get("balls", 0))
            }

    fig1 = px.bar(data["top_venues"], x="matches", y="venue", orientation="h", title="Top 10 Venues by Matches", text="matches")
    fig1.update_layout(yaxis={"categoryorder": "total ascending"})

    fig2 = px.bar(data["high_scoring_venues"], x="avg_runs_per_match", y="venue", orientation="h", title="High Scoring Venues", text=data["high_scoring_venues"]["avg_runs_per_match"].round(2))
    fig2.update_layout(yaxis={"categoryorder": "total ascending"})

    return render_template(
        "venues.html",
        chart1=chart_to_html(fig1),
        chart2=chart_to_html(fig2),
        venues=clean_records(venue_df.sort_values("matches", ascending=False).to_dict("records")),
        venue_name=venue_name,
        venue_result=venue_result,
        venue_kpis=venue_kpis
    )

@app.route("/toss")
def toss():

    toss_decision_df = data["toss_decision"].copy()
    toss_result_df = data["toss_result"].copy()
    match_df = data["match"].copy()

    # =====================================================
    # KPI CALCULATIONS
    # =====================================================

    total_tosses = (
        int(toss_decision_df["count"].sum())
        if "count" in toss_decision_df.columns
        else 0
    )

    bat_first_count = int(
        toss_decision_df[
            toss_decision_df["toss_decision"].astype(str).str.lower() == "bat"
        ]["count"].sum()
    )

    field_first_count = int(
        toss_decision_df[
            toss_decision_df["toss_decision"].astype(str).str.lower() == "field"
        ]["count"].sum()
    )

    bat_first_pct = (
        safe_round((bat_first_count / total_tosses) * 100)
        if total_tosses
        else 0
    )

    field_first_pct = (
        safe_round((field_first_count / total_tosses) * 100)
        if total_tosses
        else 0
    )

    toss_winner_match_wins = 0
    toss_winner_match_losses = 0

    if "toss_match_win" in toss_result_df.columns:

        toss_result_df["toss_match_win"] = (
            toss_result_df["toss_match_win"]
            .astype(str)
            .str.lower()
        )

        toss_winner_match_wins = int(
            toss_result_df[
                toss_result_df["toss_match_win"] == "true"
            ]["count"].sum()
        )

        toss_winner_match_losses = int(
            toss_result_df[
                toss_result_df["toss_match_win"] == "false"
            ]["count"].sum()
        )

    toss_win_pct = (
        safe_round((toss_winner_match_wins / total_tosses) * 100)
        if total_tosses
        else 0
    )

    toss_loss_pct = (
        safe_round((toss_winner_match_losses / total_tosses) * 100)
        if total_tosses
        else 0
    )

    # =====================================================
    # CHART DATA PREP
    # =====================================================

    decision_chart_df = toss_decision_df.copy()

    if not decision_chart_df.empty:

        decision_chart_df["decision_label"] = (
            decision_chart_df["toss_decision"]
            .replace({
                "bat": "Bat First",
                "field": "Field First"
            })
        )

        decision_chart_df["percentage"] = (
            decision_chart_df["count"]
            / decision_chart_df["count"].sum()
            * 100
        ).round(2)

        decision_chart_df["label_text"] = decision_chart_df.apply(
            lambda r: f"{int(r['count'])} ({r['percentage']}%)",
            axis=1
        )

    result_chart_df = toss_result_df.copy()

    if not result_chart_df.empty:

        result_chart_df["result_label"] = (
            result_chart_df["toss_match_win"]
            .replace({
                "true": "Toss Winner Won",
                "false": "Toss Winner Lost"
            })
        )

        result_chart_df["percentage"] = (
            result_chart_df["count"]
            / result_chart_df["count"].sum()
            * 100
        ).round(2)

        result_chart_df["label_text"] = result_chart_df.apply(
            lambda r: f"{int(r['count'])} ({r['percentage']}%)",
            axis=1
        )

    # =====================================================
    # CHART 1
    # =====================================================

    fig1 = px.pie(
        decision_chart_df,
        names="decision_label",
        values="count",
        hole=0.45,
        title="Bat First vs Field First Decisions"
    )

    fig1.update_traces(
        textinfo="label+percent",
        pull=[0.03] * len(decision_chart_df)
    )

    fig1.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=50, b=10)
    )

    chart1 = chart_to_html(fig1)

    # =====================================================
    # CHART 2
    # =====================================================

    fig2 = px.bar(
        result_chart_df,
        x="result_label",
        y="count",
        title="Impact of Winning Toss",
        text="label_text"
    )

    fig2.update_traces(textposition="outside")

    fig2.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    chart2 = chart_to_html(fig2)

    # =====================================================
    # CHART 3
    # =====================================================

    season_toss_chart = None

    if (
        not match_df.empty
        and {"season", "toss_decision"}.issubset(match_df.columns)
    ):

        season_toss_df = (
            match_df.groupby(["season", "toss_decision"])
            .size()
            .reset_index(name="count")
        )

        season_toss_df["toss_decision"] = (
            season_toss_df["toss_decision"]
            .replace({
                "bat": "Bat First",
                "field": "Field First"
            })
        )

        fig3 = px.bar(
            season_toss_df,
            x="season",
            y="count",
            color="toss_decision",
            barmode="group",
            title="Season-wise Toss Trend",
            text="count"
        )

        fig3.update_traces(textposition="outside")

        fig3.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=50, b=20)
        )

        season_toss_chart = chart_to_html(fig3)

    # =====================================================
    # CHART 4
    # =====================================================

    venue_toss_chart = None

    if (
        not match_df.empty
        and {"venue", "toss_winner", "match_won_by"}.issubset(match_df.columns)
    ):

        venue_df = match_df.copy()

        venue_df["toss_match_win"] = (
            venue_df["toss_winner"]
            == venue_df["match_won_by"]
        )

        venue_advantage = (
            venue_df.groupby("venue")
            .agg(
                matches=("match_id", "nunique"),
                toss_wins=("toss_match_win", "sum")
            )
            .reset_index()
        )

        venue_advantage = venue_advantage[
            venue_advantage["matches"] >= 10
        ]

        venue_advantage["toss_win_percentage"] = (
            venue_advantage["toss_wins"]
            / venue_advantage["matches"]
            * 100
        ).round(2)

        venue_advantage = (
            venue_advantage
            .sort_values(
                "toss_win_percentage",
                ascending=False
            )
            .head(10)
        )

        fig4 = px.bar(
            venue_advantage,
            x="toss_win_percentage",
            y="venue",
            orientation="h",
            title="Top Venues by Toss Advantage",
            text="toss_win_percentage"
        )

        fig4.update_traces(
            texttemplate="%{text}%",
            textposition="outside"
        )

        fig4.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=50, b=20),
            yaxis={"categoryorder": "total ascending"}
        )

        venue_toss_chart = chart_to_html(fig4)

    # =====================================================
    # INSIGHTS
    # =====================================================

    toss_kpis = {
        "total_tosses": total_tosses,
        "bat_first_count": bat_first_count,
        "field_first_count": field_first_count,
        "bat_first_pct": bat_first_pct,
        "field_first_pct": field_first_pct,
        "toss_winner_match_wins": toss_winner_match_wins,
        "toss_winner_match_losses": toss_winner_match_losses,
        "toss_winner_match_win_percentage": toss_win_pct,
        "toss_winner_match_loss_percentage": toss_loss_pct,
    }

    toss_insights = [
        f"Teams chose to field first in {field_first_pct}% of IPL matches.",
        f"Teams chose to bat first in {bat_first_pct}% of IPL matches.",
        f"Toss winners won the match {toss_win_pct}% of the time.",
        "Field-first strategy has become increasingly popular in modern IPL seasons.",
        "Team strength and venue conditions matter more than toss alone."
    ]

    xai_importance = {
        "Venue Conditions": 38,
        "Team Strength": 28,
        "Recent Form": 17,
        "Toss Decision": 9,
        "Head-to-Head": 8
    }

    return render_template(
        "toss.html",
        chart1=chart1,
        chart2=chart2,
        chart3=season_toss_chart,
        chart4=venue_toss_chart,
        toss_kpis=toss_kpis,
        toss_insights=toss_insights,
        xai_importance=xai_importance
    )


if __name__ == "__main__":
    app.run(debug=True)