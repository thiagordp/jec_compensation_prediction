"""
Code for prediction

@author Thiago R. Dal Pont
@date August 11, 2020
"""
import logging

from data_processing.preprocessing import process_attributes, load_attributes
from main import read_docs
from pipelines import representation
from pipelines.remove_outliers import remove_outliers
from pipelines.feature_selection import bow_feature_selection


def predicting_pipeline(inputs, dict_train_info):
    logging.info("=" * 50)
    logging.info("PREDICTING MODELS PIPELINE")

    # Carregar atributos
    dict_attributes = load_attributes(inputs)

    # Criar BOW usando transformer
    tf_transformer = dict_train_info["tf_transformer"]

    tf_inputs, tf_transformer, list_features = representation.represent_bow_tf(
        dict_inputs=inputs,
        vectorizer=tf_transformer,
        predict=True
    )

    # Feature selection using transformer

    # fs_transformer = dict_train_info["fs_transformer"]
    # dict_bow, fs_transformer = bow_feature_selection(
    #     dict_bow=tf_inputs,
    #     fs_transformer=fs_transformer
    # )

    dict_attributes_transf = dict_train_info["attrib_transformer"]

    dict_attributes, dict_attributes_transf = representation.transform_attributes(
        dict_attrib=dict_attributes,
        transformer=dict_attributes_transf
    )
    dict_final_bow = representation.append_attributes_to_bow(tf_inputs, dict_attributes)

    outliers_transf = dict_train_info["outliers_transf"]
    dict_final_bow, outliers_transf = remove_outliers(dict_final_bow, outliers_transf)

    return None
