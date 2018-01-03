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


class Interpreter(object):
    # Defines all attributes (and their default values) that will be returned by `parse`
    @staticmethod
    def default_output_attributes():
        return {"intent": {"name": "", "confidence": 0.0}, "entities": []}

    @staticmethod
    def load(model_metadata, config, plugin_factory=None, skip_valdation=False):
        # type: (Metadata, MyNLUConfig, Optional[PluginFactory], bool) -> Interpreter
        context = {}

        if plugin_factory is None:
            # If no builder is passed, every interpreter creation will result in a new builder.
            plugin_factory = PluginFactory()

        _pipeline = []

        if not skip_valdation:
            pipeline.validate_requirements(model_metadata.pipeline)

        for plugin_name in model_metadata.pipeline:
            plugin = plugin_factory.load_plugin(
                    plugin_name, model_metadata.model_dir, model_metadata, **context)
            try:
                updates = plugin.provide_context()
                if updates:
                    context.update(updates)

                _pipeline.append(plugin)
            except MissingArgumentError as e:
                raise Exception("Failed to initialize plugin '{}'. {}".format(plugin_name, e))

        return Interpreter(_pipeline, context, model_metadata)

    def __init__(self, pipeline, context, model_metadata=None):
        # type: (List[Component], Dict[Text, Any], Optional[Metadata]) -> None

        self.pipeline = pipeline
        self.context = context if context is not None else {}
        self.model_metadata = model_metadata

    def parse(self, text, time=None):
        # type: (Text) -> Dict[Text, Any]
        """Parse the input text, classify it and return an object containing its intent and entities."""

        if not text:
            # Not all plugins are able to handle empty strings. So we need to prevent that...
            # This default return will not contain all output attributes of all plugins,
            # but in the end, no one should pass an empty string in the first place.
            output = self.default_output_attributes()
            output["text"] = ""
            return output

        message = Message(text, self.default_output_attributes(), time=time)

        for p in self.pipeline:
            p.process(message, **self.context)

        output = self.default_output_attributes()
        output.update(message.as_dict(only_output_properties=True))
        return output