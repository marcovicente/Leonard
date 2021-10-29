import pickle as pkl
import pandas as pd
from datetime import date
from dateutil import parser
import os
import sys

MODEL_PATH = os.path.join(sys.path[0], "models/gaussian_nb_pca_model.pkl")
DATA_PATH = os.path.join(sys.path[0], "data/soccer_dataset.csv")
LABEL_DECODER = {0: 'AWAY_WIN', 1: 'HOME_WIN', 2: 'TIE'}


def predict_match_result(home_team_id: int, away_team_id: int,
                         match_date_str: str):

    # Reads CSV processed
    df = read_data(DATA_PATH)

    match_date = date(int(match_date_str[0:4]),
                      int(match_date_str[4:6]),
                      int(match_date_str[6:8]))

    df["date"] = df["date"].apply(lambda x: parser.parse(x).date())

    df_home = (df[(df["home_team_api_id"] == home_team_id)
               & (df["date"] < match_date)]
               .sort_values("date", ascending=False)
               .head(1)
               .reset_index(drop=True))

    if not same_season(match_date, df_home.loc[0, "date"]):
        return "The home team has less than 5 games, in this season."

    df_home = df_home[HOME_FEATURES]

    df_away = (df[(df["away_team_api_id"] == away_team_id)
               & (df["date"] < match_date)]
               .sort_values("date", ascending=False)
               .head(1)
               .reset_index(drop=True))
    if not same_season(match_date, df_away.loc[0, "date"]):
        return "The away team has less than 5 games, in this season."

    df_away = df_away[AWAY_FEATURES]

    df_input = pd.concat([df_home, df_away], axis=1)[INPUT_FEATURES]

    model = read_model(MODEL_PATH)
    pred = model.predict(df_input)
    prob = model.predict_proba(df_input).max()
    decoded_pred = LABEL_DECODER[pred[0]]

    return {"result": decoded_pred, "confidence": prob}


HOME_FEATURES = ['stage', 'year', 'buildUpPlaySpeed_home',
                 'buildUpPlayPassing_home', 'chanceCreationPassing_home',
                 'chanceCreationCrossing_home', 'chanceCreationShooting_home',
                 'defencePressure_home', 'defenceAggression_home',
                 'defenceTeamWidth_home', 'overall_rating_1',
                 'overall_rating_2',
                 'overall_rating_3', 'overall_rating_4', 'overall_rating_5',
                 'overall_rating_6', 'overall_rating_7', 'overall_rating_8',
                 'overall_rating_9', 'overall_rating_10', 'overall_rating_11',
                 'country_id_1', 'country_id_1729', 'country_id_4769',
                 'country_id_7809',
                 'country_id_10257', 'country_id_13274', 'country_id_15722',
                 'country_id_17642', 'country_id_19694', 'country_id_21518',
                 'country_id_24558',
                 'buildUpPlayPositioningClass_home_Free Form',
                 'buildUpPlayPositioningClass_home_Organised',
                 'chanceCreationPositioningClass_home_Free Form',
                 'chanceCreationPositioningClass_home_Organised',
                 'defenceDefenderLineClass_home_Cover',
                 'defenceDefenderLineClass_home_Offside Trap',
                 'home_score', 'home_score_m-1', 'home_score_m-2',
                 'home_score_m-3', 'home_score_m-4']


AWAY_FEATURES = ['buildUpPlaySpeed_away',
                 'buildUpPlayPassing_away', 'chanceCreationPassing_away',
                 'chanceCreationCrossing_away', 'chanceCreationShooting_away',
                 'defencePressure_away', 'defenceAggression_away',
                 'defenceTeamWidth_away', 'overall_rating_12',
                 'overall_rating_13',
                 'overall_rating_14', 'overall_rating_15', 'overall_rating_16',
                 'overall_rating_17', 'overall_rating_18', 'overall_rating_19',
                 'overall_rating_20', 'overall_rating_21', 'overall_rating_22',
                 'buildUpPlayPositioningClass_away_Free Form',
                 'buildUpPlayPositioningClass_away_Organised',
                 'chanceCreationPositioningClass_away_Free Form',
                 'chanceCreationPositioningClass_away_Organised',
                 'defenceDefenderLineClass_away_Cover',
                 'defenceDefenderLineClass_away_Offside Trap',
                 'away_score', 'away_score_m-1', 'away_score_m-2',
                 'away_score_m-3', 'away_score_m-4']

INPUT_FEATURES = ['stage', 'year', 'buildUpPlaySpeed_home',
                  'buildUpPlayPassing_home', 'chanceCreationPassing_home',
                  'chanceCreationCrossing_home', 'chanceCreationShooting_home',
                  'defencePressure_home', 'defenceAggression_home',
                  'defenceTeamWidth_home', 'buildUpPlaySpeed_away',
                  'buildUpPlayPassing_away', 'chanceCreationPassing_away',
                  'chanceCreationCrossing_away', 'chanceCreationShooting_away',
                  'defencePressure_away', 'defenceAggression_away',
                  'defenceTeamWidth_away', 'overall_rating_1',
                  'overall_rating_2',
                  'overall_rating_3', 'overall_rating_4', 'overall_rating_5',
                  'overall_rating_6', 'overall_rating_7', 'overall_rating_8',
                  'overall_rating_9', 'overall_rating_10', 'overall_rating_11',
                  'overall_rating_12', 'overall_rating_13',
                  'overall_rating_14',
                  'overall_rating_15', 'overall_rating_16',
                  'overall_rating_17',
                  'overall_rating_18', 'overall_rating_19',
                  'overall_rating_20',
                  'overall_rating_21', 'overall_rating_22', 'country_id_1',
                  'country_id_1729', 'country_id_4769', 'country_id_7809',
                  'country_id_10257', 'country_id_13274', 'country_id_15722',
                  'country_id_17642', 'country_id_19694', 'country_id_21518',
                  'country_id_24558',
                  'buildUpPlayPositioningClass_home_Free Form',
                  'buildUpPlayPositioningClass_home_Organised',
                  'chanceCreationPositioningClass_home_Free Form',
                  'chanceCreationPositioningClass_home_Organised',
                  'defenceDefenderLineClass_home_Cover',
                  'defenceDefenderLineClass_home_Offside Trap',
                  'buildUpPlayPositioningClass_away_Free Form',
                  'buildUpPlayPositioningClass_away_Organised',
                  'chanceCreationPositioningClass_away_Free Form',
                  'chanceCreationPositioningClass_away_Organised',
                  'defenceDefenderLineClass_away_Cover',
                  'defenceDefenderLineClass_away_Offside Trap',
                  'home_score', 'home_score_m-1', 'home_score_m-2',
                  'home_score_m-3',
                  'home_score_m-4', 'away_score', 'away_score_m-1',
                  'away_score_m-2', 'away_score_m-3', 'away_score_m-4']


def read_model(path):

    with open(path, "rb") as f:
        return pkl.load(f)


def read_data(path):

    return pd.read_csv(path)


def same_season(date1, date2):

    diff_days = abs((date1 - date2).days)

    if diff_days < 30:
        return True
    else:
        return False
