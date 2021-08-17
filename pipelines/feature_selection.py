"""
Feature Selction code

@date August 10, 2021
"""

import logging
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest, mutual_info_regression


def bow_feature_selection(dict_bow, y=None, k=None, fs_transformer=None):
    logging.info("Starting feature selection")

    logging.info("Formatting data")
    array_bow = np.array(list(dict_bow.values()))

    if fs_transformer is None:
        formatted_y = np.array(list(y))

        logging.info("Selecting")
        fs_transformer = SelectKBest(score_func=mutual_info_regression, k=k)
        fit_bow = fs_transformer.fit_transform(array_bow, formatted_y)
    else:
        fit_bow = fs_transformer.transform(array_bow)

    logging.info("Converting to list BOW")
    new_bow = [list(row) for row in fit_bow]

    del fit_bow

    for i, key in enumerate(dict_bow.keys()):
        dict_bow[key] = new_bow[i]

    logging.info("Finished Feature Selection")

    return dict_bow, fs_transformer
