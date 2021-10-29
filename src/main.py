import logger
from fastapi import FastAPI
from models.predict_model import predict_match_result

my_logger = logger.get_logger("soccer prediction")

app = FastAPI()


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
    # Predicts Soccer Match result
    try:
        prediction, confidence = predict_match_result(home,
                                                      away, match_date)
        my_logger.info("Executed")
        return {"home": home, "away": away, "match_date": match_date,
                "prediction": prediction, "confidence": confidence}
    except BaseException as err:
        my_logger.error(f"Unexpected {err=}, {type(err)=}")
