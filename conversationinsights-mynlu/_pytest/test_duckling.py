# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import utilities
from mynlu.training_data import TrainingData, Message
import logging
logger = logging.getLogger(__name__)

def test_duckling_entity_extractor(component_builder):
    _config = utilities.zh_test_conf()
    _config["duckling_dimensions"] = ["time"]
    duckling = component_builder.create_component("ner_duckling", _config)
    message = Message("今天是9月14号，明天见。")
    duckling.process(message)
    logger.info("message:{}".format(message))
    entities = message.get("entities")
    assert len(entities) == 3

    # Test duckling with a defined date
    message = Message("明天见.", time="1381536182000")  # 1381536182000 == 2013/10/12 02:03:02
    duckling.process(message)
    logger.info("message:{}".format(message))
    entities = message.get("entities")
    assert len(entities) == 1
    assert entities[0]["text"] == "明天"
    assert entities[0]["value"] == "2013-10-13T00:00:00.000Z"


def test_duckling_entity_extractor_and_synonyms(component_builder):
    _config = utilities.zh_test_conf()
    _config["duckling_dimensions"] = ["number"]
    duckling = component_builder.create_component("ner_duckling", _config)
    synonyms = component_builder.create_component("ner_synonyms", _config)
    message = Message("他离着有6公里远。")
    duckling.process(message)
    logger.info("message:{}".format(message))
    synonyms.process(message)   # checks that the synonym processor can handle entities that have int values
    logger.info("message:{}".format(message))
    assert message is not None
