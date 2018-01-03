from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import itertools
import logging
import os

import matplotlib.pyplot as plt

from mynlu.config.mynluconfig import MyNLUConfig
from mynlu.model.trainingdata import load_data
from mynlu.parsers import Interpreter
from mynlu.model import Metadata
from mynlu.evaluation import plot_intent_confusion_matrix

logger = logging.getLogger(__name__)


class Evaluator(object):
    def __init__(self, config, model_path, plugin_factory=None):
        # type: (MyNLUConfig, string, Optional[PluginFactory]) -> None
        self.config = config
        self.model_path = model_path
        self.plugin_factory = plugin_factory

    def run_intent_evaluation(self):
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import classification_report
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import f1_score
        from sklearn.metrics import precision_score
        from sklearn.utils.multiclass import unique_labels

        # get the metadata config from the package data
        test_data = load_data(self.config['data'])
        metadata = Metadata.load(self.model_path)
        interpreter = Interpreter.load(metadata, self.config, self.plugin_factory)

        test_y = [e.get("intent") for e in test_data.training_examples]

        preds = []
        for e in test_data.training_examples:
            res = interpreter.parse(e.text)
            if res.get('intent'):
                preds.append(res['intent'].get('name'))
            else:
                preds.append(None)

        logger.info("Intent Evaluation Results")
        logger.info("F1-Score:  {}".format(f1_score(test_y, preds, average='weighted')))
        logger.info("Precision: {}".format(precision_score(test_y, preds, average='weighted')))
        logger.info("Accuracy:  {}".format(accuracy_score(test_y, preds)))
        logger.info("Classification report: \n{}".format(classification_report(test_y, preds)))

        cnf_matrix = confusion_matrix(test_y, preds)
        plot_intent_confusion_matrix(cnf_matrix, classes=unique_labels(test_y, preds),
                                     title='Intent Confusion matrix', normalize=False)
        # plt.savefig("xxxx.png")
        return