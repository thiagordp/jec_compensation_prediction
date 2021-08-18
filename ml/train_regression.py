import logging
import random

from sklearn.ensemble import VotingRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import numpy as np


def transform_x_y(x_dict, y_dict):
    x = list()
    y = list()

    keys = list(x_dict.keys())
    random.shuffle(keys)
    logging.info("Keys training order %s" % str(keys[:10]))

    for key in keys:
        x.append(x_dict[key])
        y.append(y_dict[key])

    x = np.array(x)
    y = np.array(y)

    return x, y


def train_ensemble_voting(x_dict, y_dict):
    logging.info("Training regression model")

    x, y = transform_x_y(x_dict, y_dict)

    regressor = VotingRegressor(n_jobs=4, verbose=True, estimators=[
        ("mlp", MLPRegressor(hidden_layer_sizes=(256, 256, 256, 256, 256,),
                             max_iter=50,
                             early_stopping=True,
                             shuffle=True,
                             activation="relu",
                             batch_size=16)),
        ('bagging', BaggingRegressor(n_estimators=50, n_jobs=8)),
        ('xgb', xgb.XGBRegressor(n_estimators=50, max_depth=10, n_jobs=8)),
        ('gd', GradientBoostingRegressor(max_depth=10, max_leaf_nodes=100))
    ])

    regressor.fit(x, y)
    logging.info("Finished training; making predictions on train set as a simple verification")
    logging.info("Train set predictions %s" % str(list(regressor.predict(x))[:10]))

    return regressor


def predict_ensemble_voting(x_dict, regressor):
    logging.info("Prediction in test set")

    keys = list(x_dict.keys())

    x = np.array([x_dict[key] for key in keys])

    y_pred = regressor.predict(x)
    logging.info("| CASE | PREDICTION |")
    for i in range(len(keys)):
        logging.info("| %s | %f |" % (str(keys[i]), y_pred[i]))
