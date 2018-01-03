from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import

import typing
from typing import Any
from typing import Dict
from typing import List
from typing import Text

from mynlu.pipeline.plugin import Plugin
from mynlu.config.mynluconfig import MyNLUConfig
from mynlu.featurizers import Featurizer
from mynlu.tokenizers import Token
from mynlu.trainers import Message
from mynlu.trainers import TrainingData

import  logging
logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import mitie
    import numpy as np
    from builtins import str


class MitieFeaturizer(Featurizer):
    name = "intent_featurizer_mitie"

    provides = ["text_features"]

    requires = ["tokens", "mitie_feature_extractor"]

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["mitie", "numpy"]

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, MyNLUConfig, **Any) -> None

        mitie_feature_extractor = self._mitie_feature_extractor(**kwargs)
        for example in training_data.intent_examples:
            features = self.features_for_tokens(example.get("tokens"), mitie_feature_extractor)
            example.set("text_features", self._combine_with_existing_text_features(example, features))
        logger.debug("training_data:{}".format(training_data.intent_examples[0]))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        mitie_feature_extractor = self._mitie_feature_extractor(**kwargs)
        features = self.features_for_tokens(message.get("tokens"), mitie_feature_extractor)
        message.set("text_features", self._combine_with_existing_text_features(message, features))

    def _mitie_feature_extractor(self, **kwargs):
        mitie_feature_extractor = kwargs.get("mitie_feature_extractor")
        if not mitie_feature_extractor:
            raise Exception("Failed to train 'intent_featurizer_mitie'. Missing a proper MITIE feature extractor.")
        return mitie_feature_extractor

    def features_for_tokens(self, tokens, feature_extractor):
        # type: (List[Token], mitie.total_word_feature_extractor) -> np.ndarray
        import numpy as np

        vector = np.zeros(feature_extractor.num_dimensions)
        for token in tokens:
            vector += feature_extractor.get_feature_vector(token.text)
        if tokens:
            return vector / len(tokens)
        else:
            return vector
