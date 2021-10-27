from typing import Optional
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "Leonard"}


@app.get("/predict_result/{home_team_id}/{away_team_id}")
def read_item(home_team_id: int, away_team_id: int, q: Optional[str] = None):
    return {"home_team_id": home_team_id, "away_team_id": away_team_id, "q": q}
