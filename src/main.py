import logger
import time
from fastapi import FastAPI
from dataclasses import dataclass
from models.predict_model import predict_match_result, read_model, MODEL_PATH

my_logger = logger.get_logger("soccer prediction")
model = read_model(MODEL_PATH)


@dataclass
class Match:
    # TODO: implement API using dataclass as parameter
    # https://fastapi.tiangolo.com/advanced/dataclasses/
    home_team: int
    away_team: int
    match_date: str


app = FastAPI()


@app.on_event("shutdown")
def shutdown_event():
    my_logger.info("Application shutdown")


@app.get("/")
def read_root():
    """ Root Hello World

    Returns
    -------
    Hello World
    """
    return {"Hello": "Leonard ⚽"}


@app.get("/home/{home}/away/{away}/match_date/{match_date}")
def predict(home: int, away: int, match_date: str):
    """ Predicts the outcome of a given game between two Soccer Teams

    Parameters
    ----------
    home : int
        [Home Team]
    away : int
        [Away Team]
    match_date : str
        [Match Date]

    Returns
    -------
    [Outcome]
        [Prediction]
        [Confidence]

    Examples:
    -------
    Arsenal vs Man United (Winning Streak)
    http://127.0.0.1:8000/home/9825/away/10260/match_date/20141201

    Arsenal vs Man United (Loosing Streak)
    http://127.0.0.1:8000/home/9825/away/10260/match_date/20141226

    """
    # Records starting time
    start = time.process_time()

    # Predicts Soccer Match result
    try:
        prediction, confidence = predict_match_result(home,
                                                      away, match_date, model)

        # Loggs time spent
        my_logger.info("Predicted soccer game Outcome in {}"
                       .format((time.process_time() - start)))

        # Returns prediction
        return {"home": home, "away": away, "match_date": match_date,
                "prediction": prediction, "confidence": confidence}
    except BaseException as err:
        my_logger.error(f"Unexpected {err=}, {type(err)=}")
