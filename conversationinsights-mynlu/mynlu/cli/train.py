from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import argparse
import logging
import os

from mynlu.trainers.trainer import do_train

from mynlu.config.mynluconfig import MyNLUConfig

logger = logging.getLogger(__name__)


def create_argparser():
    parser = argparse.ArgumentParser(description='train a custom language parser')

    parser.add_argument('-p', '--pipeline', default=None,
                        help="Pipeline to use for the message processing.")
    parser.add_argument('-o', '--path', default=None,
                        help="Path where model files will be saved")
    parser.add_argument('-d', '--data', default=None,
                        help="File containing training data")
    parser.add_argument('-c', '--config', required=True,
                        help="NLU configuration file")
    parser.add_argument('-l', '--language', default=None, choices=['zh', 'en'],
                        help="Model and data language")
    parser.add_argument('-t', '--num_threads', default=None, type=int,
                        help="Number of threads to use during model training")
    parser.add_argument('-m', '--mitie_file', default=None,
                        help='File with mitie total_word_feature_extractor')
    return parser


def init():  # pragma: no cover
    # type: () -> MyNLUConfig
    """Combines passed arguments to create NLU config."""

    parser = create_argparser()
    args = parser.parse_args()
    config = MyNLUConfig(args.config, os.environ, vars(args))
    return config


if __name__ == '__main__':
    config = init()
    logging.basicConfig(level=config['log_level'])

    do_train(config)
    logger.info("Finished training")
