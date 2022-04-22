from typing import Tuple, Optional, List
from annotation.pipes.fastlang_detector import FastLangDetector
from annotation.pipes.lang_detector import LangDetector
from annotation.pipes.phrase_detector import PhraseDetector
from annotation.pipes.sentence_detector import SentenceDetector
from annotation.pipes.spacy_pipline import SpacyPipeline
from annotation.pipes.stanza_pipeline import StanzaPipeline
from annotation.pipes.spell_detector import SpellDetector
from spacy.language import Language

from annotation.pipes.umls_concept_detector import UMLSConceptDetector


@Language.factory("spacy_pipeline", default_config={"lang": "en",
                                                    "package": None,
                                                    "pipes": None,
                                                    "sentence_detector": False, })
def create_stanza_pipeline_component(nlp: Language, name: str, lang: str, package: str,
                                     pipes: Optional[str], sentence_detector: bool) -> SpacyPipeline:
    return SpacyPipeline(lang, package, pipes, sentence_detector)


@Language.factory("stanza_pipeline", default_config={"lang": "en",
                                                     "package": "default",
                                                     "processors": None,
                                                     "processors_packages": None,
                                                     "use_gpu": False,
                                                     "set_token_vector_hooks": False,
                                                     "attrs": ("metadata", "source_text", "preprocessed_text",
                                                               "sentence_sentiments"), })
def create_stanza_pipeline_component(nlp: Language, name: str, lang: str, package: Optional[str],
                                     processors: Optional[str], processors_packages: Optional[str], use_gpu: bool,
                                     set_token_vector_hooks: bool, attrs: Tuple[str, str, str, str]) -> StanzaPipeline:
    return StanzaPipeline(nlp, lang, package, processors, processors_packages, use_gpu, set_token_vector_hooks, attrs)


@Language.factory("fastlang_detector", default_config={"attrs": ("language", "language_score"),
                                                       "model_name": "lid.176.ftz", })
def create_lang_detector_component(nlp: Language, name: str, attrs: Tuple[str, str],
                                   model_name: str) -> FastLangDetector:
    return FastLangDetector(attrs, model_name)


@Language.factory("sentence_detector", default_config={"lang": "en"})
def create_sentence_detector_component(nlp: Language, name: str, lang: str) -> SentenceDetector:
    return SentenceDetector(lang)


@Language.factory("lang_detector", default_config={"attrs": ("language", "language_score")})
def create_lang_detector_component(nlp: Language, name: str, attrs: Tuple[str, str]) -> LangDetector:
    return LangDetector(attrs)


@Language.factory("phrase_detector", default_config={"attrs": ("phrases",)})
def create_phrase_chunker_component(nlp, name, attrs: Tuple[str]) -> PhraseDetector:
    return PhraseDetector(attrs)


@Language.factory("spell_detector", default_config={"attrs": ("spell_is_correct", "suggest_spellings", "misspellings")})
def create_spell_checker_component(nlp: Language, name: str, attrs: Tuple[str, str, str]) -> SpellDetector:
    return SpellDetector(attrs)


@Language.factory("umls_concept_detector", default_config={"quickumls_filepath": None,
                                                           "overlapping_criteria": "score",
                                                           "similarity_name": "jaccard",
                                                           "threshold": 0.85,
                                                           "window": 5,
                                                           "accepted_semtypes": None,
                                                           "best_match": True,
                                                           "keep_uppercase": False,
                                                           "attrs": ("umls_concepts",), })
def create_umls_concept_detector_component(nlp: Language, name: str, quickumls_filepath: Optional[str],
                                           overlapping_criteria: str, similarity_name: str, threshold: float,
                                           window: int,
                                           accepted_semtypes: Optional[List[str]], best_match: bool,
                                           keep_uppercase: bool,
                                           attrs: Tuple[str]) -> UMLSConceptDetector:
    return UMLSConceptDetector(nlp, quickumls_filepath, overlapping_criteria, similarity_name, threshold, window,
                               accepted_semtypes, best_match, keep_uppercase, attrs)
