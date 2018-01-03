from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import pytest

import utilities
from utilities import slowtest
from mynlu import registry
import logging
logger = logging.getLogger(__name__)

@slowtest
def test_samples(component_builder):
    #logger.info("pipeline_template:{}".format(pipeline_template))
    _conf = utilities.zh_test_conf_nlp_mitie()
    _conf["data"] = "./data/examples/mynlu/demo.json"
    logger.info("_conf:{}".format(_conf.view()))
    interpreter = utilities.interpreter_for(component_builder, _conf)
    available_intents = ["greet", "restaurant_search", "affirm", "goodbye", "None"]
    samples = [
        (
            "good bye",
            {
                'intent': 'goodbye',
                'entities': []
            }
        ),
        (
            "i am looking for an indian spot",
            {
                'intent': 'restaurant_search',
                'entities': [{"start": 20, "end": 26, "value": "indian", "entity": "cuisine"}]
            }
        )
    ]

    for text, gold in samples:
        result = interpreter.parse(text, time=None)
        logger.debug("result:{}".format(result))
        assert result['text'] == text, \
            "Wrong text for sample '{}'".format(text)
        assert result['intent']['name'] in available_intents, \
            "Wrong intent for sample '{}'".format(text)
        assert result['intent']['confidence'] >= 0, \
            "Low confidence for sample '{}'".format(text)

        # This ensures the model doesn't detect entities that are not present
        # Models on our test data set are not stable enough to require the entities to be found
        for entity in result['entities']:
            del entity["extractor"]
            assert entity in gold['entities'], \
                "Wrong entities for sample '{}'".format(text)
