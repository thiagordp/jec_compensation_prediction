"""


"""
import logging

import pandas as pd
import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder


def represent_bow_tf(dict_inputs=None):
    logging.info("Starting text representation")

    if dict_inputs is None:
        dict_inputs = dict()

    logging.info("Transforming dict to list")
    list_inputs = dict_inputs.values()

    logging.info("Fit Transform BOW")
    vectorizer = CountVectorizer(ngram_range=(1, 4), max_features=25000, min_df=1)
    tf_inputs = vectorizer.fit_transform(list_inputs).toarray()
    feature_names = vectorizer.get_feature_names()
    logging.info("Features: %s" % len(feature_names))

    dict_outputs = {}

    logging.info("Transforming to dict")
    for i, key in enumerate(dict_inputs.keys()):
        dict_outputs[key] = list(tf_inputs[i])

    logging.info("Finished representation")

    return dict_outputs, vectorizer, feature_names


def append_attributes_to_bow(dict_inputs, dict_attributes):
    logging.info("Appending Attributes to inputs")

    for key_input in tqdm.tqdm(dict_inputs.keys()):
        dict_inputs[key_input].append(dict_attributes[key_input])

    logging.info("Finished appending")

    return dict_inputs


def __process_judge(judges, distinct_judges, type_judges, distinct_type_judges):

    judges = [str(judge_name).strip().lower().replace(" ", "_").replace(".", "") for judge_name in judges]
    type_judges = [str(judge_type).strip().lower().replace(" ", "_").replace(".", "") for judge_type in type_judges]

    judges = pd.get_dummies(judges, prefix='juiz')
    type_judges = pd.get_dummies(type_judges, prefix="tipo_juiz")

    if distinct_judges is not None:
        for distinct in distinct_judges:

            if distinct not in judges.columns:
                judges[distinct] = 0

    if distinct_type_judges is not None:

        for distinct_type in distinct_type_judges:

            if distinct_type not in type_judges.columns:
                judges[distinct_type] = 0

    list_distinct_judges = set(judges.columns)
    list_distinct_type_judges = set(type_judges.columns)

    judges = [list(judge) for judge in list(judges.to_numpy())]
    type_judges = [list(type_judge) for type_judge in list(type_judges.to_numpy())]

    return judges, list_distinct_judges, type_judges, list_distinct_type_judges


def __process_has_x(feature, transf_feature):


    # If transf is None
    #   Fit the new encoder
    #   Transform the feature

    if transf_feature is None:
        transf_feature = LabelEncoder()

        transf_feature.fit(feature)

    return transf_feature.transform(feature), transf_feature


def __process_loss(arg):
    return list(), 0


def __process_time_delay(arg):
    return list(), 0


def transform_attributes(dict_inputs):
    """

    :param dict_inputs:
    :return:
        Transformed attributes
        Dict of transformers
    """

    logging.info("Transforming attributes")

    raw_data_df = pd.DataFrame.from_dict(dict_inputs, orient='index')

    raw_data_df.to_excel("test.xlsx", index=False)

    # Extract attributes
    days_list = list(raw_data_df["dia"])
    months_list = list(raw_data_df["mes"])
    years_list = list(raw_data_df["ano"])
    day_week_list = list(raw_data_df["dia_semana"])
    judges = list(raw_data_df["juiz"])
    type_judges = list(raw_data_df["tipo_juiz"])

    # Todo change to get from dict
    judges, transf_judges, type_judges, transf_type_judges = __process_judge(judges, None, type_judges, None)

    has_permanent_loss_list, has_permanent_loss_transf = __process_has_x(raw_data_df["extravio_permanente"].values, None)
    has_temporally_loss_list, has_temporally_loss_transf = __process_has_x(raw_data_df["extravio_temporario"].values, None)
    interval_loss_list, interval_loss_transf = __process_loss(raw_data_df["intevalo_extravio"].values)
    has_luggage_violation_list, has_luggage_violation_transf = __process_has_x(raw_data_df["tem_violacao_bagagem"].values, None)
    has_flight_delay_list, has_flight_delay_transf = __process_has_x(raw_data_df["tem_atraso_voo"].values, None)
    has_flight_cancellation_list, has_flight_cancellation_transf = __process_has_x(raw_data_df["tem_cancelamento_voo"].values, None)
    flight_delay_list, flight_delay_transf = __process_time_delay(raw_data_df["qtd_atraso_voo"].values)
    is_consumers_fault_list, is_consumers_fault_transf = __process_has_x(raw_data_df["culpa_consumidor"].values, None)
    has_adverse_flight_conditions_list, has_adverse_flight_conditions_transf = __process_has_x(raw_data_df["tem_condicao_adversa_voo"].values, None)
    has_no_show_list, has_no_show_transf = __process_has_x(raw_data_df["tem_no_show"].values, None)
    has_overbooking_list, has_overbooking_transf = __process_has_x(raw_data_df["tem_overbooking"].values, None)
    has_cancel_refunding_problem_list, has_cancel_refunding_transf = __process_has_x(raw_data_df["tem_cancelamento_usuario_ressarcimento"].values, None)
    has_offer_disagreement_list, has_offer_disagreement_transf = __process_has_x(raw_data_df["tem_desacordo_oferta"].values, None)

    return None, None