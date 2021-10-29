from fastapi import FastAPI
import logging

# Configure Logging
logging.basicConfig(format='%(asctime)s %(message)s')

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "Leonard ⚽"}


@app.get("/home_team/{home_team}/away_team/{away_team}/date/{date}")
def predict(home_team: int, away_team: int, date: str):
    return {"home_team": home_team, "away_team": away_team, "date": date}
