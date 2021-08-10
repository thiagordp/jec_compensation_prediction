"""
Constants for the project
"""


class Defs:
    JEC_BASE_DATASET_PATH = "/media/trdp/Arquivos/Studies/dev_phd/datasets/jec_compensation_prediction/"
    JEC_TRAIN_DATASET_PATH = "jec_train_dataset_wo_result/"
    JEC_TEST_DATASET_PATH = "jec_test_dataset_wo_result/"

    JEC_ATTRIBUTES_PROC = "data/attributes.csv"

    APPLY_STEMMING = False
    REMOVE_STOPWORDS = True
    GET_INDIVIDUAL_VALUES = True
    K_BEST_FEATURES = 500

    def __init__(self):
        pass


OPTION_FIELDS = {
    "--stemming": Defs.APPLY_STEMMING
}
