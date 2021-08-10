"""
Feature Selction code

@date August 10, 2021
"""

import logging

from sklearn.feature_selection import SelectKBest, mutual_info_regression


def bow_feature_selection(dict_bow, y, k):
    logging.info("Starting feature selection")

    # TODO: Get y values
    fs_transformer = SelectKBest(score_func=mutual_info_regression, k=k)

    fit_bow = fs_transformer.fit_transform(dict_bow.values(), y)

    new_bow = [list(row) for row in fit_bow]
    del fit_bow

    return new_bow, fs_transformer
