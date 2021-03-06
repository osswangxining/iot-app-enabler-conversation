from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from typing import Any
from typing import Dict
from typing import List
from typing import Text

from mynlu.config.mynluconfig import MyNLUConfig
from mynlu.tokenizers import Tokenizer, Token
from mynlu.pipeline.plugin import Plugin
from mynlu.trainers import Message
from mynlu.trainers import TrainingData


class WhitespaceTokenizer(Tokenizer, Plugin):
    name = "tokenizer_whitespace"

    provides = ["tokens"]

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, MyNLUConfig, **Any) -> None

        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.text))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        message.set("tokens", self.tokenize(message.text))

    def tokenize(self, text):
        # type: (Text) -> List[Token]

        words = text.split()
        running_offset = 0
        tokens = []
        for word in words:
            word_offset = text.index(word, running_offset)
            word_len = len(word)
            running_offset = word_offset + word_len
            tokens.append(Token(word, word_offset))
        return tokens
