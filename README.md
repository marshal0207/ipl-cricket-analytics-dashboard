# 🏏 Explainable AI-Based Cricket Intelligence & Match Analytics Platform

An AI-powered IPL analytics platform that goes beyond simple predictions — combining match intelligence, team/player analytics, and **Explainable AI (XAI)** to show not just *who wins*, but *why*.

🔗 **Live Demo:** [http://34.229.169.233:5000](http://34.229.169.233:5000)

---

## 📌 Overview

Most cricket prediction tools give you a single number and stop there. This platform is built around **transparency** — every prediction comes with a breakdown of the statistical factors that drove it, using model feature importance translated into plain-language cricket insights.

## ✨ Features

- **Match Prediction** — Random Forest model predicts the winner between any two IPL teams with win probability and confidence score
- **Explainable AI Insights** — feature importance bars and AI-generated reasoning explain *why* a prediction was made
- **Team Analytics** — win rates, head-to-head records, recent form, best/toughest opponents
- **Player Analytics** — batting and bowling career stats with searchable lookup
- **Venue Analytics** — ground-level scoring trends and venue-specific win rates
- **Toss Analysis** — quantifies how much (or little) winning the toss actually matters
- **Season Analytics** — year-on-year scoring trends across IPL history
- **Match Scorecard & Schedule** — full historical match browsing with innings breakdowns

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Machine Learning | Scikit-Learn (Random Forest) |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| Frontend | HTML, CSS, JavaScript, Jinja2 |
| Deployment | AWS EC2 |
| Version Control | Git, GitHub, Git LFS |

## 🚀 Running Locally

```bash
# Clone the repo
git clone https://github.com/marshal0207/ipl-cricket-analytics-dashboard.git
cd ipl-cricket-analytics-dashboard

# Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
python flask_app.py
```

Then open `http://127.0.0.1:5000` in your browser.

> **Note:** This repo uses Git LFS for large CSV/model files. Run `git lfs install` before cloning if you don't already have LFS set up.

## 📂 Project Structure

```
ipl-cricket-analytics-dashboard/
├── flask_app.py              # Main Flask application
├── train_winner_model.py     # Model training script
├── requirements.txt
├── data/                     # Cleaned IPL datasets
├── models/                   # Trained model & encoders (.pkl)
├── static/
│   ├── css/                  # Stylesheet
│   └── images/                # Team logos & assets
└── templates/                 # Jinja2 HTML templates
```

## 🧠 Why Explainable AI?

Black-box predictions erode trust. This platform surfaces the model's reasoning — feature importance, head-to-head context, and recent form — so users can evaluate *how* a prediction was reached, not just accept the output blindly.

## 📈 Model Notes

The prediction model is a Random Forest Classifier trained on historical IPL match data (2008–2024), using team win percentages, head-to-head records, recent form, venue, toss, and season as features. Given the inherent high variance of T20 cricket, the model's accuracy is consistent with published cricket-analytics benchmarks for pre-match statistical prediction.

## 👤 Author

**Marshal Godhani** 

---

⭐ If you found this project interesting, consider giving it a star!