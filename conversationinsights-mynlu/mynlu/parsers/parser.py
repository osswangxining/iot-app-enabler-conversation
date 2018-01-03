from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import datetime

import glob
import json
import logging
import os
import tempfile
import io

from builtins import object
from typing import Text
from future.utils import PY3

from concurrent.futures import ProcessPoolExecutor
from twisted.internet.defer import Deferred, maybeDeferred
from twisted.logger import jsonFileLogObserver, Logger

from mynlu import utils
from mynlu.pipeline.plugin import PluginFactory
from mynlu.model import InvalidModelError, Metadata
from mynlu.parsers import Interpreter
from mynlu.model.metadata import TrainingModelMetadata

logger = logging.getLogger(__name__)


class DataParser(object):
    DEFAULT_MODEL_NAME = "default"

    def __init__(self, config, plugin_factory):
        self._training_processes = config['max_training_processes'] if config['max_training_processes'] > 0 else 1
        self.config = config
        self.responses = DataParser._create_query_logger(config['response_log'])
        self._trainings_queued = 0
        self.model_dir = config['path']
        self.token = config['token']
        self.plugin_factory = plugin_factory if plugin_factory else PluginFactory(cache_enabled=True)
        self.model_store = self.__create_model_store()
        self.pool = ProcessPoolExecutor(self._training_processes)

    @staticmethod
    def _create_query_logger(response_log_dir):
        """Creates a logger that will persist incomming queries and their results."""

        if response_log_dir:
            # We need to generate a unique file name, even in multiprocess environments
            timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            log_file_name = "nlu_log-{}-{}.log".format(timestamp, os.getpid())
            response_logfile = os.path.join(response_log_dir, log_file_name)
            # Instantiate a standard python logger, which we are going to use to log requests
            utils.create_dir_for_file(response_logfile)
            query_logger = Logger(observer=jsonFileLogObserver(io.open(response_logfile, 'a', encoding='utf8')),
                                  namespace='query-logger')
            # Prevents queries getting logged with parent logger --> might log them to stdout
            logger.info("Logging requests to '{}'.".format(response_logfile))
            return query_logger
        else:
            # If the user didn't provide a logging directory, we wont log!
            logger.info("Logging of requests is disabled. (No 'request_log' directory configured)")
            return None

    def __create_model_store(self):
        # Fallback for users that specified the model path as a string and hence only want a single default model.
        if type(self.config.server_model_dirs) is Text:
            model_dict = {self.DEFAULT_MODEL_NAME: self.config.server_model_dirs}
        elif self.config.server_model_dirs is None:
            model_dict = self.__search_for_models()
        else:
            model_dict = self.config.server_model_dirs

        model_store = {}

        for alias, model_path in list(model_dict.items()):
            try:
                logger.info("Loading model '{}'...".format(model_path))
                model_store[alias] = self.__interpreter_for_model(model_path)
            except Exception as e:
                logger.exception("Failed to load model '{}'. Error: {}".format(model_path, e))
        if not model_store:
            meta = Metadata({"pipeline": ["intent_classifier_keyword"]}, "")
            interpreter = Interpreter.load(meta, self.config, self.plugin_factory)
            model_store[self.DEFAULT_MODEL_NAME] = interpreter
        return model_store

    def __search_for_models(self):
        models = {}
        for metadata_path in glob.glob(os.path.join(self.config.path, '*/metadata.json')):
            model_name = os.path.basename(os.path.dirname(metadata_path))
            models[model_name] = model_name
        return models

    def __interpreter_for_model(self, model_path):
        metadata = TrainingModelMetadata.read_model_metadata(model_path, self.config)
        return Interpreter.load(metadata, self.config, self.plugin_factory)

    def extract(self, data):
        _data = {}
        _data["text"] = data["q"][0] if type(data["q"]) == list else data["q"]
        if not data.get("model"):
            _data["model"] = "default"
        elif type(data["model"]) == list:
            _data["model"] = data["model"][0]
        else:
            _data["model"] = data["model"]
        _data['time'] = data["time"] if "time" in data else None
        return _data

    def parse(self, data):
        alias = data.get("model") or self.DEFAULT_MODEL_NAME
        if alias not in self.model_store:
            try:
                self.model_store[alias] = self.__interpreter_for_model(model_path=alias)
            except Exception as e:
                raise InvalidModelError("No model found with alias '{}'. Error: {}".format(alias, e))

        model = self.model_store[alias]
        logger.info("model:{}".format(model))
        logger.info("data:{}".format(data))
        response = model.parse(data['text'], data.get('time', None))
        logger.info("response:{}".format(response))
        if self.responses:
            self.responses.info(user_input=response, model=alias)
        return response