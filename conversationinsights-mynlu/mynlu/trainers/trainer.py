from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import argparse
import datetime
import logging
import os
import copy
import typing
from typing import Text
from typing import Tuple
from mynlu import pipeline
from mynlu.pipeline.plugin import PluginFactory, Plugin
from mynlu.model.trainingdata import load_data
from mynlu.parsers import Interpreter
from mynlu.model import Metadata
from mynlu.config.mynluconfig import MyNLUConfig
from typing import Optional
from mynlu.utils import create_dir

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from mynlu.persistor import Persistor


def do_train_in_worker(config):
    # type: (MyNLUConfig) -> Text
    """Loads the trainer and the data and runs the training of the specified model in a subprocess."""
    logger.info("creating one do_train_in_worker....")
    _, _, persisted_path = do_train(config)
    return persisted_path


def do_train(config, plugin_factory=None):
    # type: (MyNLUConfig, Optional[PluginFactory]) -> Tuple[Trainer, Interpreter, Text]
    """Loads the trainer and the data and runs the training of the specified model."""

    # Ensure we are training a model that we can save in the end
    # WARN: there is still a race condition if a model with the same name is trained in another subprocess
    logger.info("creating one trainer....")
    trainer = Trainer(config, plugin_factory)
    logger.info("creating one persistor....")
    persistor = create_persistor(config)
    training_data = load_data(config['data'])
    logger.info("creating one interpreter....")
    interpreter = trainer.train(training_data)
    persisted_path = trainer.persist(config['path'], persistor, model_name=config['name'])

    return trainer, interpreter, persisted_path


def create_persistor(config):
    # type: (MyNLUConfig) -> Optional[Persistor]
    """Create a remote persistor to store the model if the configuration requests it."""

    persistor = None
    if "bucket_name" in config:
        from mynlu.persistor import get_persistor
        persistor = get_persistor(config)

    return persistor


class Trainer(object):
    SUPPORTED_LANGUAGES = ["zh"]

    def __init__(self, config, plugin_factory=None, skip_validation=False):
        # type: (MyNLUConfig, Optional[PluginFactory], bool) -> None

        self.config = config
        self.skip_validation = skip_validation
        self.training_data = None  # type: Optional[TrainingData]
        self.pipeline = []
        if plugin_factory is None:
            # If no builder is passed, every interpreter creation will result in a new builder.
            plugin_factory = PluginFactory()

        if not self.skip_validation:
            pipeline.validate_requirements(config.pipeline)

        for plugin_name in config.pipeline:
            plugin = plugin_factory.create_plugin(plugin_name, config)
            self.pipeline.append(plugin)

    def train(self, data):
        # type: (TrainingData) -> Interpreter
        """Trains the underlying pipeline by using the provided training data."""

        self.training_data = data

        context = {}  # type: Dict[Text, Any]

        for p in self.pipeline:
            updates = p.provide_context()
            if updates:
                context.update(updates)

        if not self.skip_validation:
            pipeline.validate_arguments(self.pipeline, context)

        working_data = copy.deepcopy(data)  # data gets modified internally during the training - hence the copy

        for i, plugin in enumerate(self.pipeline):
            logger.info("Starting to training plugin {}".format(plugin.name))
            plugin.prepare_partial_processing(self.pipeline[:i], context)
            updates = plugin.train(working_data, self.config, **context)
            logger.info("Finished training plugin.")
            if updates:
                context.update(updates)

        return Interpreter(self.pipeline, context)

    def persist(self, path, persistor=None, model_name=None):
        # type: (Text, Optional[Persistor], Text) -> Text

        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        metadata = {
            "language": self.config["language"],
            "pipeline": [plugin.name for plugin in self.pipeline],
        }

        if model_name is None:
            dir_name = os.path.join(path, "model_" + timestamp)
        else:
            dir_name = os.path.join(path, model_name)

        create_dir(dir_name)

        if self.training_data:
            metadata.update(self.training_data.persist(dir_name))

        for p in self.pipeline:
            update = p.persist(dir_name)
            if update:
                metadata.update(update)

        metadata = Metadata(metadata, dir_name)
        metadata.persist(dir_name)

        if persistor is not None:
            persistor.save_tar(dir_name)
        logger.info("Successfully saved model into '{}'".format(os.path.abspath(dir_name)))
        return dir_name