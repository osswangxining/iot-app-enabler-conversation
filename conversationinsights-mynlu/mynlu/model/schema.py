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

logger = logging.getLogger(__name__)


def mynlu_data_schema():
    training_example_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "intent": {"type": "string"},
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "number"},
                        "end": {"type": "number"},
                        "value": {"type": "string"},
                        "entity": {"type": "string"}
                    },
                    "required": ["start", "end", "entity"]
                }
            }
        },
        "required": ["text"]
    }

    regex_feature_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "pattern": {"type": "string"},
        }
    }

    return {
        "type": "object",
        "properties": {
            "mynlu_data": {
                "type": "object",
                "properties": {
                    "regex_features": {
                        "type": "array",
                        "items": regex_feature_schema
                    },
                    "common_examples": {
                        "type": "array",
                        "items": training_example_schema
                    }
                }
            }
        },
        "additionalProperties": False
    }


def validate_mynlu_data(data):
    # type: (Dict[Text, Any]) -> None
    """Validate training data format to ensure proper training. Raises exception on failure."""
    from jsonschema import validate
    from jsonschema import ValidationError

    try:
        validate(data, mynlu_data_schema())
    except ValidationError as e:
        e.message += \
            ". Failed to validate training data, make sure your data is valid. "
        raise e
