"""

@author Thiago R. Dal Pont
@date August 11, 2020
"""
import logging

from sklearn.ensemble import IsolationForest


def remove_outliers(dict_inputs, transformer=None):
    logging.info("Starting to remove outliers step")
    values = list(dict_inputs.values())

    if transformer is None:
        transformer = IsolationForest(n_jobs=8, contamination=0.1, verbose=0)

        logging.info("Training outliers detector")
        is_outlier_list = transformer.fit_predict(values, None)
    else:
        is_outlier_list = transformer.predict(values)

    dict_outputs = dict()
    logging.info("Removing outliers")
    for index_key, key_input in enumerate(dict_inputs.keys()):
        if is_outlier_list[index_key] > 0:
            dict_outputs[key_input] = dict_inputs[key_input]

    logging.info("Finished removing outliers")

    return dict_outputs, transformer
