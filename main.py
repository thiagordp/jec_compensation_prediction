"""
Prediction of compensation values of legal judgments frmo JEC.

@author Thiago Raulino Dal Pont
@date Aug 06, 2021
"""
import glob
import json
import logging
import sys

import pandas as pd

from pipelines import training, predicting
from data_processing.preprocessing import process_attributes
from util.constants import Defs, OPTION_FIELDS


def setup_logging():
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filename='jec_prediction.log',
                        level=logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)


def read_docs(path_to_docs, list_ids):
    logging.info("Reading content from docs in %s" % path_to_docs)

    docs = dict()

    list_path_files = glob.glob(path_to_docs + "*")
    logging.info("\tFound %d docs", len(list_path_files))

    for path_file in list_path_files:
        with open(path_file, "r", encoding="utf-8") as fp:
            text = fp.read()
            dict_key = path_file.replace(path_to_docs, "").replace(".txt", "")

            if int(dict_key) in list_ids:
                docs[dict_key] = text

    logging.info("List of documents:")

    with open("data/dict_docs.json", "w+", encoding="utf-8") as fp:
        fp.write(json.dumps(docs, indent=4, sort_keys=True, ensure_ascii=False))

    return docs


def setup_options(args):
    for i in range(len(args)):
        arg = args[i]
        try:
            field = OPTION_FIELDS[arg]
            field = args[i + 1]
        except:
            pass


def main(args):
    setup_logging()
    # setup_options(args)
    logging.info("-" * 50)
    logging.info("PREDICTION OF COMPENSATION VALUES FROM JEC")

    list_ids = process_attributes(
        file_path="data/regression_data_attributes.xlsx",
        sheet_name=0
    )

    docs = read_docs("data/train_data/", list_ids)

    dict_train_info = training.training_pipeline(docs)
    # ict_train_info = None
    # ------------------------------------------------- #
    logging.info("="*50)
    list_ids = process_attributes(
        file_path="data/regression_data_attributes.xlsx",
        sheet_name=1,
        ignore_zero=False
    )
    docs = read_docs("data/test_data/", list_ids)

    predicting.predicting_pipeline(docs, dict_train_info)


if __name__ == "__main__":
    main(sys.argv[1:])
