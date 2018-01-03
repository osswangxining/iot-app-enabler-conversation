from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals, print_function

import typing
from typing import Any
from typing import Dict
from typing import List
from typing import Text

from mynlu.extractors import EntityExtractor
from mynlu.trainers import Message

if typing.TYPE_CHECKING:
    from spacy.tokens.doc import Doc


class SpacyEntityExtractor(EntityExtractor):
    name = "ner_spacy"

    provides = ["entities"]

    requires = ["spacy_doc"]

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        extracted = self.add_extractor_name(self.extract_entities(message.get("spacy_doc")))
        message.set("entities", message.get("entities", []) + extracted, add_to_output=True)

    def extract_entities(self, doc):
        # type: (Doc) -> List[Dict[Text, Any]]

        entities = [
            {
                "entity": ent.label_,
                "value": ent.text,
                "start": ent.start_char,
                "end": ent.end_char
            }
            for ent in doc.ents]
        return entities
