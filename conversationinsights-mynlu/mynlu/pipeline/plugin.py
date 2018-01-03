from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import logging
import os
from collections import defaultdict

import typing
from builtins import object
import inspect

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Text
from typing import Tuple

from mynlu.config.mynluconfig import MyNLUConfig
from mynlu.trainers import Message
from mynlu.pipeline import MissingArgumentError

if typing.TYPE_CHECKING:
    from mynlu.trainers import TrainingData
    from mynlu.model import Metadata

logger = logging.getLogger(__name__)


class PluginFactory(object):
    def __init__(self, cache_enabled=True):
        self.cache_enabled = cache_enabled
        self.cached_plugins = {}

    def __add_to_cache(self, key, plugin):
        if key is not None and self.cache_enabled:
            self.cached_plugins[key] = plugin
            logger.info("Plugin {} is cached using key {}.".format(plugin.name, key))

    def __get_from_cache(self, plugin_name, metadata):
        from mynlu import registry

        plugin = registry.get_plugin(plugin_name)
        key = plugin.cache_key(metadata)
        if key is not None and self.cache_enabled and key in self.cached_plugins:
            return self.cached_plugins[key], key

        return None, key

    def create_plugin(self, plugin_name, config):
        try:
            from mynlu.model import Metadata
            metadata = Metadata(config.as_dict(), None)
            plugin, key = self.__get_from_cache(plugin_name, metadata)
            if plugin is None:
                from mynlu import registry
                plugin = registry.create_plugin_by_name(plugin_name, config)
                self.__add_to_cache(key, plugin)
            return plugin
        except MissingArgumentError as e:
            raise Exception("Failed to create plugin '{}' with the following exception: {}".format(plugin_name, e))

    def load_plugin(self, plugin_name, model_dir, metadata, **context):
        try:
            plugin, key = self.__get_from_cache(plugin_name, metadata)
            if plugin is None:
                from mynlu import registry
                plugin = registry.load_plugin_by_name(plugin_name, model_dir, metadata, plugin, **context)
            return plugin
        except MissingArgumentError as e:
            raise Exception("Failed to load plugin '{}' with the following exception: {}".format(plugin_name, e))


class Plugin(object):
    """A plugin is a message processing unit in a pipeline, which can be executed sequentially (If a plugin comes
       first in a pipeline, its methods will be called first).

      It can include the initialization, training, processing, persisting and loading the plugins.

      During the processing (as well as the training, persisting and initialization), the plugin in the pipeline
      can pass information to others. This information is passed to other plugins by providing attributes to the
      pipeline context. The pipeline context contains all the information of the previous plugins which can be used
      to do its own processing. For example, a featurizer plugin can provide features that are used by another
      plugin down the pipeline to do intent classification."""

    # the plugin name - should be unique in the pipeline
    name = ""

    # Defines what attributes the pipeline plugin will provide when called. The listed attributes
    # should be set by this plugin on the message object during test and train, e.g.
    # ```message.set("entities", [...])```
    provides = []

    # Which attributes on a message are required by this plugin. e.g. if a classifier plugin requires contains
    # "text_features", it means that the previous plugin in this pipeline should provide "text_features".
    requires = []

    def __init__(self):
        self.partial_processing_pipeline = None
        self.partial_processing_context = None

    def __getstate__(self):
        state = self.__dict__.copy()
        if "partial_processing_pipeline" in state:
            del state["partial_processing_pipeline"]
        if "partial_processing_context" in state:
            del state["partial_processing_context"]
        return state

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return []

    @classmethod
    def load(cls, model_dir=None, model_metadata=None, cached_plugin=None, **kwargs):
        # type: (Text, Metadata, Optional[Component], **Any) -> Component
        return cached_plugin if cached_plugin else cls()

    def persist(self, model_dir):
        # type: (Text) -> Optional[Dict[Text, Any]]
        pass

    @classmethod
    def create(cls, config):
        # type: (MyNLUConfig) -> Component
        """Creates this plugin (e.g. before a training is started).
        Method can access all configuration parameters."""
        return cls()

    def provide_context(self):
        # type: () -> Optional[Dict[Text, Any]]
        """Initialize this plugin for a new pipeline

        This function will be called before the training is started and before the first message is processed using
        the interpreter. The plugin gets the opportunity to add information to the context that is passed through
        the pipeline during training and message parsing.

        It's mostly used to initialize framework environments like MITIE and spacy
        (e.g. loading word vectors for the pipeline)."""
        pass

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, MyNLUConfig, **Any) -> None
        """THe plugin can rely on any context attribute created by a call to `pipeline_init` of any plugin and
        on any context attributes created by a call to `train` of plugins previous to this one."""
        pass

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        pass

    @classmethod
    def cache_key(cls, model_metadata):
        # type: (Metadata) -> Optional[Text]
        return None

    def prepare_partial_processing(self, pipeline, context):
        """Sets the pipeline and context used for partial processing.
           The pipeline should be a list of plugins that are previous to this one in the pipeline and
            have already finished their training (and can therefore be safely used to process messages)."""
        self.partial_processing_pipeline = pipeline
        self.partial_processing_context = context

    def partially_process(self, message):
        """Allows the plugin to process messages during training
            The passed message will be processed by all plugins previous to this one in the pipeline."""
        if self.partial_processing_context is not None:
            for plugin in self.partial_processing_pipeline:
                plugin.process(message, **self.partial_processing_context)
        else:
            logger.info("Failed to run partial processing due to missing pipeline.")
        return message