from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import json
import logging

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Text

from mynlu import utils
from mynlu.tokenizers import Tokenizer
from mynlu.trainers import TrainingData, Message
from mynlu.model.schema import validate_mynlu_data

logger = logging.getLogger(__name__)


def load_data(resource_name, fformat=None):
    # type: (Text, Optional[Text]) -> TrainingData
    """Loads training data from disk. If no format is provided, the format will be guessed based on the files."""
    try:
        files = utils.recursively_find_files(resource_name)
        logger.info("Training data format at {} is {}".format(resource_name, "mynlu"))
        return load_mynlu_data(files[0])
    except ValueError as e:
        raise ValueError("Invalid training data file / folder specified. {}".format(e))


def get_entity_synonyms_dict(synonyms):
    # type: (List[Dict]) -> Dict
    """build entity_synonyms dictionary"""
    entity_synonyms = {}
    for s in synonyms:
        if "value" in s and "synonyms" in s:
            for synonym in s["synonyms"]:
                entity_synonyms[synonym] = s["value"]
    return entity_synonyms


def load_mynlu_data(filename):
    # type: (Text) -> TrainingData
    """Loads training data stored in the NLU data format."""

    with io.open(filename, encoding="utf-8-sig") as f:
        data = json.loads(f.read())
    validate_mynlu_data(data)

    common = data['mynlu_data'].get("common_examples", list())
    regex_features = data['mynlu_data'].get("regex_features", list())
    synonyms = data['mynlu_data'].get("entity_synonyms", list())

    entity_synonyms = get_entity_synonyms_dict(synonyms)


    training_examples = []
    for e in common:
        data = {}
        if e.get("intent"):
            data["intent"] = e["intent"]
        if e.get("entities") is not None:
            data["entities"] = e["entities"]
        training_examples.append(Message(e["text"], data))

    return TrainingData(training_examples, entity_synonyms, regex_features)