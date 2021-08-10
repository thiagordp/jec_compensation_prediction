"""
Code for training the models
"""
import logging
from processing_data import preprocessing


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


    logging.info("------------------------------------------------------------------------")
    # logging.info("                             Feature Selection                          ")

    logging.info("------------------------------------------------------------------------")
    # logging.info("                             Addition of AELE                           ")

    logging.info("------------------------------------------------------------------------")
    # logging.info("                              Remove Outliers                           ")

    logging.info("------------------------------------------------------------------------")
    # logging.info("                                  Training                              ")

    return outputs
