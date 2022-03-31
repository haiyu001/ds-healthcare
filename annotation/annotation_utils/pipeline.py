from typing import Optional, Any
from annotation.tokenization.base_tokenizer import SpacyBaseTokenizer, StanzaBaseTokenizer
from annotation.tokenization.normalizer import Normalizer
from annotation.tokenization.preprocessor import Preprocessor
from annotation.tokenization.tokenizer import MetaTokenizer
from annotation.annotation_utils.annotation import load_blank_nlp
from annotation.pipes.factories import *

NLP_MODEL = None


def get_nlp_model(lang: str,
                  spacy_package: str,
                  text_meta_config: Optional[Dict[str, List[str]]] = None,
                  preprocessor_config: Optional[Dict[str, bool]] = None,
                  stanza_base_tokenizer_package: Optional[Union[str, Dict[str, str]]] = None,
                  normalizer_config: Optional[Dict[str, Any]] = None,
                  spacy_pipeline_config: Optional[Dict[str, Any]] = None,
                  stanza_pipeline_config: Optional[Dict[str, Any]] = None,
                  custom_pipes_config: Optional[List[Tuple[str, Dict[str, Any]]]] = None) -> Language:
    """
    This function is used to get global nlp model
    :param lang: model language
    :param spacy_package: spacy blank nlp package name
    :param text_meta_config: None if all records are text string, otherwise set this config for json string
    :param preprocessor_config: None if don't apply preprocessor, {} if use default preprocessor config
    :param stanza_base_tokenizer_package: None if use spacy base tokenizer otherwise use stanza base tokenizer
    :param normalizer_config: None if annotation don't apply normalizer otherwise set this config for normalizer
    :param spacy_pipeline_config: None if don't use spacy pipeline, {} if use default spacy pipeline config
    :param stanza_pipeline_config: None if don't use stanza pipeline, {} if use default stanza pipeline config
    :param custom_pipes_config: None if don't apply custom pipes, otherwise [(pipe_name, pipe_config), ...}]
    :return: global spacy nlp model
    """

    global NLP_MODEL

    if NLP_MODEL is None:

        # create blank nlp
        nlp = load_blank_nlp(lang, spacy_package)

        # add nlp pipeline
        if spacy_pipeline_config is not None and stanza_pipeline_config is not None:
            raise Exception("Only one pipeline can be added.")
        elif spacy_pipeline_config is None and stanza_pipeline_config is None:
            raise Exception("One pipeline must be added.")
        elif spacy_pipeline_config is not None:
            nlp.add_pipe("spacy_pipeline", config=spacy_pipeline_config)
        else:
            nlp.add_pipe("stanza_test", config=stanza_pipeline_config)

        # add custom pipes
        if custom_pipes_config:
            for pipe_name, pipe_config in custom_pipes_config:
                nlp.add_pipe(pipe_name, config=pipe_config)

        # set nlp tokenizer
        base_tokenizer = StanzaBaseTokenizer(nlp, lang, stanza_base_tokenizer_package) \
            if stanza_base_tokenizer_package else SpacyBaseTokenizer(nlp)

        preprocessor = normalizer = None
        if preprocessor_config is not None:
            preprocessor = Preprocessor(**preprocessor_config)
        if normalizer_config is not None:
            normalizer = Normalizer(nlp, **normalizer_config)

        nlp.tokenizer = MetaTokenizer(base_tokenizer, preprocessor, normalizer, **text_meta_config) \
            if text_meta_config is not None else MetaTokenizer(base_tokenizer, preprocessor, normalizer)

        NLP_MODEL = nlp

    return NLP_MODEL
