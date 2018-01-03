from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import datetime
import io
import json
import logging
import os

import copy
from builtins import object
from builtins import str
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Text

import mynlu
from mynlu import pipeline
from mynlu.config.mynluconfig import MyNLUConfig
from mynlu.pipeline import MissingArgumentError
from mynlu.trainers import TrainingData, Message
from mynlu.utils import create_dir
from mynlu.pipeline.plugin import Plugin, PluginFactory
from mynlu import pipeline

logger = logging.getLogger(__name__)


class InvalidModelError(Exception):
    """Raised when a model failed to load.

    Attributes:
        message -- explanation of why the model is invalid
    """

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class Metadata(object):
    """Captures all necessary information about a model to load it and prepare it for usage."""

    @staticmethod
    def load(model_dir):
        # type: (Text) -> 'Metadata'
        """Loads the metadata from a models directory."""
        try:
            with io.open(os.path.join(model_dir, 'metadata.json'), encoding="utf-8") as f:
                data = json.loads(f.read())
            return Metadata(data, model_dir)
        except Exception as e:
            raise InvalidModelError("Failed to load model metadata from '{}'. {}".format(
                    os.path.abspath(os.path.join(model_dir, 'metadata.json')), e))

    def __init__(self, metadata, model_dir):
        # type: (Dict[Text, Any], Optional[Text]) -> None

        self.metadata = metadata
        self.model_dir = model_dir

    def get(self, property_name, default=None):
        return self.metadata.get(property_name, default)

    @property
    def language(self):
        # type: () -> Optional[Text]
        """Language of the underlying model"""

        return self.get('language')

    @property
    def pipeline(self):
        # type: () -> List[Text]
        """Names of the processing pipeline elements."""

        return self.get('pipeline', [])

    def persist(self, model_dir):
        # type: (Text) -> None
        """Persists the metadata of a model to a given directory."""

        metadata = self.metadata.copy()

        metadata.update({
            "trained_at": datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
            "nlu_version": mynlu.__version__,
        })

        with io.open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
            f.write(str(json.dumps(metadata, indent=4)))

