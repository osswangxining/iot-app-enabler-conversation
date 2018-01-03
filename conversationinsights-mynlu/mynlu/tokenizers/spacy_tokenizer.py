from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import typing
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Text

from mynlu.config.mynluconfig import MyNLUConfig
from mynlu.tokenizers import Tokenizer, Token
from mynlu.pipeline.plugin import Plugin
from mynlu.trainers import Message
from mynlu.trainers import TrainingData

if typing.TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens.doc import Doc


class SpacyTokenizer(Tokenizer, Plugin):
    name = "tokenizer_spacy"

    provides = ["tokens"]

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, MyNLUConfig, **Any) -> None

        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.get("spacy_doc")))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        message.set("tokens", self.tokenize(message.get("spacy_doc")))

    def tokenize(self, doc):
        # type: (Doc) -> List[Token]

        return [Token(t.text, t.idx) for t in doc]
