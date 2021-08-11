"""


"""
import logging

from sklearn.feature_extraction.text import CountVectorizer


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
