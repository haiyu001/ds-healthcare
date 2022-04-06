from typing import Union, Dict, Tuple, List, Optional
from annotation.pipes.fastlang_detector import FastLangDetector
from annotation.pipes.lang_detector import LangDetector
from annotation.pipes.phrase_detector import PhraseDetector
from annotation.pipes.sentence_detector import SentenceDetector
from annotation.pipes.spacy_pipline import SpacyPipeline
from annotation.pipes.stanza_pipeline import StanzaPipeline
from spacy.language import Language


@Language.factory("spacy_pipeline", default_config={"lang": "en",
                                                    "package": None,
                                                    "exclude": None,
                                                    "sentence_detector": False})
def create_stanza_pipeline_component(nlp: Language, name: str, lang: str, package: str,
                                     exclude: Optional[str], sentence_detector: bool) -> SpacyPipeline:
    return SpacyPipeline(lang, package, exclude, sentence_detector)


@Language.factory("stanza_pipeline", default_config={"lang": "en",
                                                     "package": "default",
                                                     "processors": None,
                                                     "processors_packages": None,
                                                     "use_gpu": False,
                                                     "set_token_vector_hooks": False,
                                                     "attrs": ("metadata", "source_text", "sentence_sentiments")})
def create_stanza_pipeline_component(nlp: Language, name: str, lang: str, package: str,
                                     processors: Optional[str], processors_packages: Optional[str], use_gpu: bool,
                                     set_token_vector_hooks: bool, attrs: Tuple[str, str, str]) -> StanzaPipeline:
    return StanzaPipeline(nlp, lang, package, processors, processors_packages, use_gpu, set_token_vector_hooks, attrs)


@Language.factory("fastlang_detector", default_config={"attrs": ("language", "language_score"),
                                                       "model_name": "lid.176.ftz"})
def create_lang_detector_component(nlp: Language, name: str, attrs: Tuple[str, str], model_name: str) -> FastLangDetector:
    return FastLangDetector(attrs, model_name)


@Language.factory("lang_detector", default_config={"attrs": ("language", "language_score")})
def create_lang_detector_component(nlp: Language, name: str, attrs: Tuple[str, str]) -> LangDetector:
    return LangDetector(attrs)


@Language.factory("sentence_detector", default_config={"lang": "en"})
def create_sentence_detector_component(nlp: Language, name: str, lang: str) -> SentenceDetector:
    return SentenceDetector(lang)


@Language.factory("phrase_detector", default_config={"attrs": ("phrases",)})
def create_phrase_chunker_component(nlp, name, attrs: Tuple[str]) -> PhraseDetector:
    return PhraseDetector(attrs)
