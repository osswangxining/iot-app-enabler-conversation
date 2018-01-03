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
from typing import Type

from mynlu.classifiers.keyword_intent_classifier import KeywordIntentClassifier
from mynlu.classifiers.mitie_intent_classifier import MitieIntentClassifier
from mynlu.classifiers.sklearn_intent_classifier import SklearnIntentClassifier
from mynlu.extractors.duckling_extractor import DucklingExtractor
from mynlu.extractors.entity_synonyms import EntitySynonymMapper
from mynlu.extractors.mitie_entity_extractor import MitieEntityExtractor
from mynlu.extractors.spacy_entity_extractor import SpacyEntityExtractor
from mynlu.extractors.crf_entity_extractor import CRFEntityExtractor
from mynlu.featurizers.mitie_featurizer import MitieFeaturizer
from mynlu.featurizers.ngram_featurizer import NGramFeaturizer
from mynlu.featurizers.regex_featurizer import RegexFeaturizer
from mynlu.featurizers.spacy_featurizer import SpacyFeaturizer
from mynlu.model import Metadata
from mynlu.tokenizers.mitie_tokenizer import MitieTokenizer
from mynlu.tokenizers.spacy_tokenizer import SpacyTokenizer
from mynlu.tokenizers.jieba_tokenizer import JiebaTokenizer
from mynlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from mynlu.nlp.mitie_nlp import MitieNLP
from mynlu.nlp.spacy_nlp import SpacyNLP

if typing.TYPE_CHECKING:
    from mynlu.config.mynluconfig import MyNLUConfig

all_plugins = [
    SpacyNLP, MitieNLP,
    SpacyEntityExtractor, MitieEntityExtractor, DucklingExtractor, CRFEntityExtractor,
    EntitySynonymMapper,
    SpacyFeaturizer, MitieFeaturizer, NGramFeaturizer, RegexFeaturizer,
    MitieTokenizer, SpacyTokenizer, WhitespaceTokenizer, JiebaTokenizer,
    SklearnIntentClassifier, MitieIntentClassifier, KeywordIntentClassifier,
]

registered_plugins = {
    plugin.name: plugin for plugin in all_plugins}

pipeline_zh_templates = {"jieba_mitie_sklearn": [
        "nlp_mitie",
        "tokenizer_jieba",
        "ner_mitie",
        "ner_synonyms",
        "intent_entity_featurizer_regex",
        "intent_featurizer_mitie",
        "intent_classifier_sklearn"
    ]}

registered_pipeline_templates = {
    # just for testing
     "all_plugins": [
        "nlp_spacy",
        "nlp_mitie",
        "tokenizer_whitespace",
        "tokenizer_jieba",
        "tokenizer_mitie",
        "tokenizer_spacy",
        "intent_featurizer_mitie",
        "intent_featurizer_spacy",
        "intent_featurizer_ngrams",
        "intent_entity_featurizer_regex",
        "ner_mitie",
        "ner_crf",
        "ner_spacy",
        "ner_duckling",
        "ner_synonyms",
        "intent_classifier_keyword",
        "intent_classifier_sklearn",
        "intent_classifier_mitie"
    ],
    "jieba_mitie_sklearn": [
        "nlp_mitie",
        "tokenizer_jieba",
        "ner_mitie",
        "ner_synonyms",
        "intent_entity_featurizer_regex",
        "intent_featurizer_mitie",
        "intent_classifier_sklearn"
    ]

}

# registered_pipeline_templates.update("jieba_mitie_sklearn", pipeline_zh_templates["jieba_mitie_sklearn"])


def get_plugin(plugin_name):
    # type: (Text) -> Optional[Type[Plugin]]
    if plugin_name not in registered_plugins:
        raise Exception("Failed to find plugin for '{}'. ".format(plugin_name))
    return registered_plugins[plugin_name]


def load_plugin_by_name(plugin_name, model_dir, metadata, cached_plugins, **kwargs):
    # type: (Text, Text, Metadata, Optional[Plugin], **Any) -> Optional[Plugin]
    _plugin = get_plugin(plugin_name)
    return _plugin.load(model_dir, metadata, cached_plugins, **kwargs)


def create_plugin_by_name(plugin_name, config):
    # type: (Text, MyNLUConfig) -> Optional[Plugin]
    _plugin = get_plugin(plugin_name)
    return _plugin.create(config)
