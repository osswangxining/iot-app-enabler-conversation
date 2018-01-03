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
from mynlu.evaluation.evaluator import Evaluator

logger = logging.getLogger(__name__)


def create_argparser():
    parser = argparse.ArgumentParser(description='evaluate a trained NLU pipeline')

    parser.add_argument('-d', '--data', default=None, help="file containing evaluation data")
    parser.add_argument('-m', '--model', required=True, help="path to model")
    parser.add_argument('-c', '--config', required=True, help="config file")
    return parser


if __name__ == '__main__':
    parser = create_argparser()
    args = parser.parse_args()
    nlu_config = MyNLUConfig(args.config, os.environ, vars(args))
    logging.basicConfig(level=nlu_config['log_level'])


    evaluator = Evaluator(nlu_config, args.model)
    evaluator.run_intent_evaluation()
    logger.info("Finished evaluation")
