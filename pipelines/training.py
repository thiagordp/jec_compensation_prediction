"""
Code for training the models
"""
import logging
from data_processing import preprocessing, representation


def training_pipeline(inputs):
    """
    Training the pipeline
    :param inputs: List of documents to use for training
    :return: The models
    """
    outputs = dict()

    logging.info("------------------------------------------------------------------------")
    processed_inputs = preprocessing.pre_processing(inputs)

    logging.info("------------------------------------------------------------------------")
    # logging.info("                            Text Representation                         ")
    tf_inputs, tf_transformer, list_features = representation.represent_bow_tf(processed_inputs)

    outputs["tf_transformer"] = tf_transformer
    outputs["tf_inputs"] = tf_inputs
    outputs["tf_features"] = list_features

    logging.info("------------------------------------------------------------------------")


    logging.info("------------------------------------------------------------------------")
    # logging.info("                             Addition of AELE                           ")

    logging.info("------------------------------------------------------------------------")
    # logging.info("                              Remove Outliers                           ")

    logging.info("------------------------------------------------------------------------")
    # logging.info("                                  Training                              ")

    return outputs
