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


from mynlu.config.mynluconfig import MyNLUConfig
from mynlu.model import Metadata

logger = logging.getLogger(__name__)

class TrainingModelMetadata(object):
    @staticmethod
    def default_model_metadata():
        return {
            "language": None,
        }

    @staticmethod
    def read_model_metadata(model_dir, config):
        if model_dir is None:
            data = TrainingModelMetadata.default_model_metadata()
            return Metadata(data, model_dir)
        else:
            if not os.path.isabs(model_dir):
                model_dir = os.path.join(config['path'], model_dir)

            # download model from remote
            if not os.path.isdir(model_dir):
                TrainingModelMetadata.load_model_from_cloud(model_dir, config)

            return Metadata.load(model_dir)

    @staticmethod
    def load_model_from_cloud(model_dir, config):
        try:
            from mynlu.persistor import get_persistor
            p = get_persistor(config)
            if p is not None:
                p.fetch_and_extract('{0}.tar.gz'.format(os.path.basename(model_dir)))
            else:
                raise RuntimeError("Unable to initialize persistor")
        except Exception as e:
            logger.warning("Using default interpreter, couldn't fetch model: {}".format(e))