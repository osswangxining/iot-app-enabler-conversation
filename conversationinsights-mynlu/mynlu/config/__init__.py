from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import os

from typing import Text

DEFAULT_CONFIG_LOCATION = "config_jieba_mitie_sklearn.json"

DEFAULT_CONFIG = {
    "name": None,
    "config": DEFAULT_CONFIG_LOCATION,
    "data": None,
    "emulate": None,
    "language": "zh",
    "log_file": None,
    "log_level": 'INFO',
    "mitie_file": os.path.join("data", "total_word_feature_extractor_zh.dat"),
    "spacy_model_name": None,
    "num_threads": 1,
    "max_training_processes": 1,
    "path": "./models",
    "port": 5000,
    "server_model_dirs": None,
    "token": None,
    "cors_origins": [],
    "max_number_of_ngrams": 7,
    "pipeline": ["nlp_mitie",
        "tokenizer_jieba",
        "ner_mitie",
        "ner_synonyms",
        "intent_entity_featurizer_regex",
        "intent_featurizer_mitie",
        "intent_classifier_sklearn"],
    "response_log": "logs",
    "aws_endpoint_url": None,
    "duckling_dimensions": None,
    "ner_crf": {
        "BILOU_flag": True,
        "features": [
            ["low", "title", "upper", "pos", "pos2"],
            ["bias", "low", "word3", "word2", "upper", "title", "digit", "pos", "pos2", "pattern"],
            ["low", "title", "upper", "pos", "pos2"]],
        "max_iterations": 50,
        "L1_c": 1,
        "L2_c": 1e-3
    },
    "intent_classifier_sklearn": {
        "C": [1, 2, 5, 10, 20, 100],
        "kernel": "linear"
    }
}


class InvalidConfigError(ValueError):
    def __init__(self, message):
        # type: (Text) -> None
        super(InvalidConfigError, self).__init__(message)