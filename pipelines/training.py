"""
Code for training the models
"""
import logging
from data_processing import preprocessing
from data_processing.preprocessing import load_attributes
from pipelines import representation
from pipelines.feature_selection import bow_feature_selection
from pipelines.remove_outliers import remove_outliers
from util.constants import Defs


def training_pipeline(inputs):
    """
    Training the pipeline
    :param inputs: List of documents to use for training
    :return: The models
    """

    outputs = dict()
    dict_compensations = dict()
    logging.info("="*50)
    logging.info("TRAINING MODELS PIPELINE")
    logging.info("-" * 50)
    dict_attributes = load_attributes(inputs)

    for key in dict_attributes.keys():
        dict_compensations[key] = dict_attributes[key]["indenizacao"]

    outputs["attributes"] = dict_attributes

    logging.info("-" * 50)

    #processed_inputs = preprocessing.pre_processing(inputs)

    logging.info("-" * 50)
    # logging.info("                            Text Representation                         ")
    tf_inputs, tf_transformer, list_features = representation.represent_bow_tf(inputs)

    outputs["tf_transformer"] = tf_transformer
    outputs["tf_features"] = list_features

    logging.info("-" * 50)
    # dict_bow, fs_transformer = bow_feature_selection(tf_inputs, dict_compensations.values(), Defs.K_BEST_FEATURES)
    #
    # outputs["tf_bow"] = dict_bow
    # outputs["fs_transformer"] = fs_transformer

    logging.info("-" * 50)
    dict_attributes, dict_attributes_transf = representation.transform_attributes(dict_attributes)

    dict_final_bow = representation.append_attributes_to_bow(tf_inputs, dict_attributes)

    logging.info("-" * 50)

    dict_final_bow, outliers_transf = remove_outliers(dict_final_bow)

    input("...")
    # TODO: Training
    logging.info("-" * 50)
    # logging.info("                                  Training                              ")

    return outputs
