from fastapi import FastAPI
from models.predict_model import predict_match_result
import logging

# Configure Logging
logging.basicConfig(format='%(asctime)s %(message)s')

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "Leonard ⚽"}


@app.get("/home_team/{home_team}/away_team/{away_team}/match_date/{match_date}")
def predict(home_team: int, away_team: int, match_date: str):
    # Predicts Soccer Match result
    prediction, confidence = predict_match_result(home_team,
                                                  away_team, match_date)

    # http://127.0.0.1:8000/home_team/9825/away_team/10260/match_date/20141212
    # Arsenal vs Man United
    
    return {"home_team": home_team, "away_team": away_team,
            "match_date": match_date}
