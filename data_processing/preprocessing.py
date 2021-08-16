"""

"""
import logging
import re
from datetime import datetime

import nltk
from matplotlib import pyplot as plt
import tqdm
from nltk import word_tokenize, RSLPStemmer
from nltk.corpus import stopwords

from util.constants import *
import pandas as pd
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')


def process_attributes():
    logging.info("Processing attributes")

    attributes_df = pd.read_csv(Defs.JEC_BASE_DATASET_PATH + "regression_data_attributes.csv",
                                usecols=[
                                    "Sentença",
                                    "Julgamento",
                                    "Valor individual do dano moral",
                                    "Data do Julgamento",
                                    "Julgador(a)",
                                    "Tipo Julgador(a)",
                                    "Extravio Definitivo",
                                    "Extravio Temporário",
                                    "Intervalo do Extravio (dias)",
                                    "Violação (furto, avaria)",
                                    "Cancelamento (sem realocação)/Alteração de destino",
                                    "Atraso (com realocação)",
                                    "Intervalo do Atraso (horas:minutos)",
                                    "Culpa exclusiva do consumidor",
                                    "Condições climáticas desfavoráveis/Fechamento aeroporto",
                                    "No Show",
                                    "Overbooking",
                                    "Cancelamento pelo consumidor e problemas com o reembolso",
                                    "Descumprimento de oferta (assento)",
                                ])
    attributes_df.dropna(subset=["Julgamento"], inplace=True)
    attributes_df.dropna(subset=["Atraso (com realocação)"], inplace=True)

    attributes_df.sort_values('Valor individual do dano moral')

    attributes_df = attributes_df[attributes_df["Data do Julgamento"].notna()]
    attributes_df = attributes_df[attributes_df["Data do Julgamento"].notnull()]

    logging.info("Starting attributes data_processing")
    final_data = list()

    for index, row in tqdm.tqdm(attributes_df.iterrows()):

        num_judgement = row["Sentença"]
        jec_class = row["Julgamento"]
        date = row["Data do Julgamento"]

        format_string = "%d/%m/%Y"
        date = datetime.strptime(date, format_string)
        year = date.year
        month = date.month
        day = date.day
        weekday = date.weekday()
        judge = row["Julgador(a)"]
        type_judge = row["Tipo Julgador(a)"]

        has_permanent_loss = row["Extravio Definitivo"]
        has_temporally_loss = row["Extravio Temporário"]
        interval_loss = row["Intervalo do Extravio (dias)"]
        has_luggage_violation = row["Violação (furto, avaria)"]
        has_flight_cancellation = row["Cancelamento (sem realocação)/Alteração de destino"]
        has_flight_delay = row["Atraso (com realocação)"]
        flight_delay = row["Intervalo do Atraso (horas:minutos)"]
        is_consumers_fault = row["Culpa exclusiva do consumidor"]
        has_adverse_flight_conditions = row["Condições climáticas desfavoráveis/Fechamento aeroporto"]
        has_no_show = row["No Show"]
        has_overbooking = row["Overbooking"]
        has_cancel_user_refunding_problem = row["Cancelamento pelo consumidor e problemas com o reembolso"]
        has_offer_disagreement = row["Descumprimento de oferta (assento)"]

        if Defs.GET_INDIVIDUAL_VALUES:
            indenizacao = float(
                str(row["Valor individual do dano moral"]).replace("R$ ", "").replace(".", "").replace(",", "."))
        else:
            indenizacao = float(
                str(row["Valor total do dano moral"]).replace("R$ ", "").replace(".", "").replace(",", "."))

        final_data.append([
            int(num_judgement),
            jec_class,
            year,
            month,
            day,
            weekday,
            judge.strip(),
            type_judge,
            indenizacao,
            has_permanent_loss,
            has_temporally_loss,
            interval_loss,
            has_luggage_violation,
            has_flight_delay,
            has_flight_cancellation,
            flight_delay,
            is_consumers_fault,
            has_adverse_flight_conditions,
            has_no_show,
            has_overbooking,
            has_cancel_user_refunding_problem,
            has_offer_disagreement
        ])

    final_df = pd.DataFrame(data=final_data,
                            columns=[
                                "judgement",
                                "jec_class",
                                "ano",
                                "mes",
                                "dia",
                                "dia_semana",
                                "juiz",
                                "tipo_juiz",
                                "indenizacao",
                                "extravio_permanente",
                                "extravio_temporario",
                                "intevalo_extravio",
                                "tem_violacao_bagagem",
                                "tem_atraso_voo",
                                "tem_cancelamento_voo",
                                "qtd_atraso_voo",
                                "culpa_consumidor",
                                "tem_condicao_adversa_voo",
                                "tem_no_show",
                                "tem_overbooking",
                                "tem_cancelamento_usuario_ressarcimento",
                                "tem_desacordo_oferta"
                            ])

    file_path = Defs.JEC_ATTRIBUTES_PROC
    final_df = final_df.loc[final_df['indenizacao'] > 1]
    final_df.to_csv(file_path, index=False)
    final_df.to_excel(file_path.replace(".csv", ".xlsx"), index=False)

    logging.info("Attributes info:")
    logging.info("\tMean %.2f" % np.mean(final_df["indenizacao"]))
    logging.info("\tStd  %.2f" % np.std(final_df["indenizacao"]))
    logging.info("\tMin  %.2f" % np.min(final_df["indenizacao"]))
    logging.info("\tMax  %.2f" % np.max(final_df["indenizacao"]))

    plt.grid(axis='y', alpha=0.3)
    n, bins, patches = plt.hist(x=final_df["indenizacao"], bins=7, color='#0504aa',
                                alpha=0.7, rwidth=0.85)

    list_ids = list(final_df["judgement"])
    plt.savefig("data/hist_attributes.pdf")

    logging.info("Finished data_processing attributes")

    return list_ids


def load_attributes(inputs):
    logging.info("Loading attributes")

    dict_outputs = dict()

    attributes_df = pd.read_csv("data/attributes.csv")

    for key in tqdm.tqdm(inputs.keys()):

        filtered_df = attributes_df.loc[attributes_df["judgement"] == int(key)]

        if len(filtered_df.index) == 0:
            logging.error("Found no attributes for judgment %s" % key)
            continue

        dict_attr = filtered_df.to_dict('records')[0]
        dict_outputs[key] = dict_attr

    logging.info("Finished loading attributes")
    # TODO: merge inputs and attr
    return dict_outputs


def remove_stopwords(text):
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if word not in stopwords.words('portuguese')]
    text = " ".join(tokens_without_sw)

    return text


def apply_stemming(text):
    text_tokens = word_tokenize(text)
    stemmer = RSLPStemmer()

    text_tokens = [stemmer.stem(token) for token in text_tokens]
    text = " ".join(text_tokens)

    return text


def clear_text(text):
    text = text.lower()
    text = text.replace("\n", " ").replace("\t", " ")
    text = text.replace("http://www ", " ")

    text = re.sub("-+", " ", text)
    text = re.sub("\.+", " ", text)
    text = text.replace("nbsp", " ")
    text = text.replace(u'\ufeff', '')
    # Symbols

    for symb in "()[]{}!?\"§_/,-“”‘’–'º•|<>$#*@:;":
        text = text.replace(symb, " ")

    for symb in ".§ºª°":
        text = text.replace(symb, " ")

    for letter in "bcdfghjklmnpqrstvwxyz":
        text = text.replace(" " + letter + " ", " ")

    for i in range(5):
        text = text.replace("  ", " ")

    return text


def pre_processing(dict_list_documents):
    logging.info("Starting preprocessing")
    processed_dict = dict()

    for key in tqdm.tqdm(dict_list_documents.keys()):
        text = dict_list_documents[key]

        text = clear_text(text)

        if Defs.REMOVE_STOPWORDS:
            text = remove_stopwords(text)
        if Defs.APPLY_STEMMING:
            text = apply_stemming(text)
        processed_dict[key] = text

    del dict_list_documents
    logging.info("Finished preprocessing")

    return processed_dict
