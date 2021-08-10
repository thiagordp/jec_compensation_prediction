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

from pipelines import training
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
    logging.info(args)
    logging.info("-" * 50)
    logging.info("PREDICTION OF COMPENSATION VALUES FROM JEC")

    list_ids = process_attributes()

    docs = read_docs(Defs.JEC_BASE_DATASET_PATH + Defs.JEC_TRAIN_DATASET_PATH, list_ids)

    training.training_pipeline(docs)


if __name__ == "__main__":
    main(sys.argv[1:])
