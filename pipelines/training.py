"""
Code for training the models
"""
import logging
from data_processing import preprocessing
from data_processing.preprocessing import load_attributes
from pipelines import representation
from pipelines.feature_selection import bow_feature_selection
from util.constants import Defs


def training_pipeline(inputs):
    """
    Training the pipeline
    :param inputs: List of documents to use for training
    :return: The models
    """
    outputs = dict()
    dict_compensations = dict()

    logging.info("-" * 50)
    dict_attributes = load_attributes(inputs)

    for key in dict_attributes.keys():
        dict_compensations[key] = dict_attributes[key]["indenizacao"]

    outputs["attributes"] = dict_attributes

    logging.info("-" * 50)

    processed_inputs = preprocessing.pre_processing(inputs)

    logging.info("-" * 50)
    # logging.info("                            Text Representation                         ")
    tf_inputs, tf_transformer, list_features = representation.represent_bow_tf(processed_inputs)

    outputs["tf_transformer"] = tf_transformer
    outputs["tf_inputs"] = tf_inputs
    outputs["tf_features"] = list_features

    logging.info("-" * 50)
    bow_feature_selection(tf_inputs, dict_compensations.values(), Defs.K_BEST_FEATURES)

    # TODO: continuar aqui
    logging.info("-" * 50)

    # logging.info("                             Addition of AELE                           ")

    logging.info("-" * 50)
    # logging.info("                              Remove Outliers                           ")

    logging.info("-" * 50)
    # logging.info("                                  Training                              ")

    return outputs
