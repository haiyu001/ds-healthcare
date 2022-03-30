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
                  preprocessor_config: Dict[str, bool] = {},
                  text_meta_config: Dict[str, List[str]] = {},
                  normalizer_config: Optional[Dict[str, Any]] = None,
                  spacy_pipeline_config: Optional[Dict[str, Any]] = None,
                  stanza_pipeline_config: Optional[Dict[str, Any]] = None,
                  custom_pipes_config: Optional[List[Tuple[str, Dict[str, Any]]]] = None) -> Language:

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
            base_tokenizer = SpacyBaseTokenizer(lang)
            nlp.add_pipe("spacy_pipeline", config=spacy_pipeline_config)
        else:
            base_tokenizer = StanzaBaseTokenizer(nlp.vocab, lang, stanza_pipeline_config.get("package", "default"))
            nlp.add_pipe("stanza_test", config=stanza_pipeline_config)

        # add custom pipes
        for pipe_name, pipe_config in custom_pipes_config:
            nlp.add_pipe(pipe_name, config=pipe_config)

        # set nlp tokenizer
        preprocessor = Preprocessor(**preprocessor_config)
        normalizer = Normalizer(nlp.vocab, **normalizer_config) if normalizer_config else None
        nlp.tokenizer = MetaTokenizer(base_tokenizer, preprocessor, normalizer, **text_meta_config)

        NLP_MODEL = nlp

    return NLP_MODEL