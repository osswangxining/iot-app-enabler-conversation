from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import tempfile

import pytest
from builtins import object
from mynlu.data_router import TrainingHelper

from mynlu import registry
from mynlu.cli.train import do_train
from mynlu.config import MyNLUConfig
from mynlu.parsers import Interpreter

logger = logging.getLogger(__name__)
slowtest = pytest.mark.slowtest


def base_test_conf(pipeline_template):
    return MyNLUConfig(cmdline_args={
        'response_log': temp_log_file_dir(),
        'port': 5022,
        "pipeline": registry.registered_pipeline_templates.get(pipeline_template, []),
        "path": tempfile.mkdtemp(),
        "data": "./data/test/demo-mynlu-small.json"
    })


def zh_test_conf():
    return MyNLUConfig(cmdline_args={
        'response_log': temp_log_file_dir(),
        'port': 5023,
        "pipeline": ["nlp_mitie",
            "tokenizer_jieba",
            "ner_mitie",
            "ner_synonyms",
            "intent_entity_featurizer_regex",
            "intent_featurizer_mitie",
            "intent_classifier_sklearn"],
        "path": tempfile.mkdtemp(),
        "language": "zh",
        "mitie_file": "./data/total_word_feature_extractor_zh.dat",
        "data": "./data/test/demo-mynlu-small.json"
    })


def zh_test_conf_nlp_mitie():
    return MyNLUConfig(cmdline_args={
        'response_log': temp_log_file_dir(),
        'port': 5023,
        "pipeline": ["nlp_mitie",
                     "tokenizer_jieba",
                     "ner_mitie",
                     "ner_synonyms",
                     "intent_entity_featurizer_regex",
                     "intent_featurizer_mitie",
                     "intent_classifier_sklearn"],
        "path": tempfile.mkdtemp(),
        "language": "zh",
        "mitie_file": "./data/total_word_feature_extractor_zh.dat",
        "data": "./data/test/demo-mynlu-small.json"
    })


def write_file_config(file_config):
    with tempfile.NamedTemporaryFile("w+", suffix="_tmp_config_file.json", delete=False) as f:
        f.write(json.dumps(file_config))
        f.flush()
        return f


def interpreter_for(component_builder, config):
    (trained, path) = run_train(config, component_builder)
    interpreter = load_interpreter_for_model(config, path, component_builder)
    return interpreter


def temp_log_file_dir():
    temp_log_file = tempfile.mkdtemp(suffix="_mynlu_logs")
    logger.info("temp_log_file:{}".format(temp_log_file))
    return temp_log_file


def run_train(config, component_builder):
    (trained, _, path) = do_train(config, component_builder)
    return trained, path


def load_interpreter_for_model(config, persisted_path, component_builder):
    metadata = TrainingHelper.read_model_metadata(persisted_path, config)
    return Interpreter.load(metadata, config, component_builder)


class ResponseTest(object):
    def __init__(self, endpoint, expected_response, payload=None):
        self.endpoint = endpoint
        self.expected_response = expected_response
        self.payload = payload
