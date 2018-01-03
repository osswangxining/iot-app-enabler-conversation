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

from mynlu.pipeline.plugin import PluginFactory
from mynlu.config.mynluconfig import MyNLUConfig
from mynlu.trainers.trainer import do_train_in_worker

logger = logging.getLogger(__name__)


def deferred_from_future(future):
    """Converts a concurrent.futures.Future object to a twisted.internet.defer.Deferred obejct.
    See: https://twistedmatrix.com/pipermail/twisted-python/2011-January/023296.html
    """
    d = Deferred()

    def callback(future):
        e = future.exception()
        if e:
            d.errback(e)
            return
        d.callback(future.result())

    future.add_done_callback(callback)
    return d


class TrainingHelper(object):
    def __init__(self, config, plugin_factory):
        self._training_processes = config['max_training_processes'] if config['max_training_processes'] > 0 else 1
        self.config = config
        # self.responses = DataRouter._create_query_logger(config['response_log'])
        self._trainings_queued = 0
        self.model_dir = config['path']
        self.token = config['token']
        # self.emulator = self.__create_emulator()
        self.plugin_factory = plugin_factory if plugin_factory else PluginFactory(cache_enabled=True)
        # self.model_store = self.__create_model_store()
        self.pool = ProcessPoolExecutor(self._training_processes)

    def __del__(self):
        """Terminates workers pool processes"""
        self.pool.shutdown()

    def shutdown(self):
        """Public wrapper over the internal __del__ function"""
        self.__del__()

    def _add_training_to_queue(self):
        """Adds a new training process to the list of running processes."""
        self._trainings_queued += 1

    def _remove_training_from_queue(self):
        """Decreases the ongoing trainings count by one"""
        self._trainings_queued -= 1

    def get_status(self):
        # TODO need to collect the status of all the training jobs in the future
        models = glob.glob(os.path.join(self.model_dir, '*'))
        models = [model for model in models if os.path.isfile(os.path.join(model, "metadata.json"))]
        return {
            "available_models": models,
            "trainings_queued": self._trainings_queued,
            "training_workers": self._training_processes
        }

    def start_train_process(self, data, config_values):
        if PY3:
            f = tempfile.NamedTemporaryFile("w+", suffix="_training_data.json", delete=False, encoding="utf-8")
            f.write(data)
        else:
            f = tempfile.NamedTemporaryFile("w+", suffix="_training_data.json", delete=False)
            f.write(data.encode("utf-8"))
        f.close()

        _config = self.config.as_dict()
        for key, val in config_values.items():
            _config[key] = val
        _config["data"] = f.name
        train_config = MyNLUConfig(cmdline_args=_config)
        logger.info("Preparing for the new training.....")

        def training_callback(model_path):
            self._remove_training_from_queue()
            return os.path.basename(os.path.normpath(model_path))

        self._add_training_to_queue()
        logger.info("Submitting the training worker.....")
        result = self.pool.submit(do_train_in_worker, train_config)
        result = deferred_from_future(result)
        result.addCallback(training_callback)

        return result
