from typing import Union, Dict, Tuple, List
from annotation.pipes.language_detector import LangDetector
from annotation.pipes.phrase_detector import PhraseDetector
from annotation.pipes.sentence_detector import SentenceDetector
from annotation.pipes.spacy_pipline import SpacyPipeline
from annotation.pipes.stanza_pipeline import StanzaPipeline
from spacy.language import Language


@Language.factory("spacy_pipeline", default_config={"lang": "en",
                                                    "package": "en_core_web_md-3.2.0",
                                                    "exclude": [],
                                                    "sentence_detector": False})
def create_stanza_pipeline_component(nlp: Language, name: str, lang: str, package: str,
                                     exclude: List[str], sentence_detector: bool) -> SpacyPipeline:
    return SpacyPipeline(lang, package, exclude, sentence_detector)


@Language.factory("stanza_pipeline", default_config={"lang": "en",
                                                     "package": "default",
                                                     "processors": {},
                                                     "use_gpu": False})
def create_stanza_pipeline_component(nlp: Language, name: str, lang: str, package:str,
                                     processors: Union[str, Dict[str, str]], use_gpu: bool) -> StanzaPipeline:
    return StanzaPipeline(nlp, lang, package, processors, use_gpu)


@Language.factory("language_detector", default_config={"attrs": ("language", "language_score"),
                                                       "model_name": "lid.176.ftz"})
def create_lang_detector_component(nlp: Language, name: str, attrs: Tuple[str, str], model_name: str) -> LangDetector:
    return LangDetector(attrs, model_name)


@Language.factory("sentence_detector", default_config={"lang": "en"})
def create_sentence_detector_component(nlp: Language, name: str, lang: str) -> SentenceDetector:
    return SentenceDetector(lang)


@Language.factory("phrase_detector", default_config={"attrs": ("phrases",)})
def create_phrase_chunker_component(nlp, name, attrs: Tuple[str]) -> PhraseDetector:
    return PhraseDetector(attrs)
