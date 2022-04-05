import warnings
from typing import Optional, Any
from annotation.tokenization.base_tokenizer import SpacyBaseTokenizer, StanzaBaseTokenizer
from annotation.annotation_utils.annotate_util import load_blank_nlp, DEFAULT_SPACY_PACKAGE, get_stanza_load_list
from annotation.tokenization.normalizer import Normalizer
from annotation.tokenization.preprocessor import Preprocessor
from annotation.tokenization.tokenizer import MetaTokenizer
from annotation.pipes.factories import *
from utils.log_util import get_logger
import spacy

NLP_MODEL = None

logger = get_logger()


def get_nlp_model(use_gpu: bool = False,
                  lang: str = "en",
                  spacy_package: Optional[str] = None,
                  text_meta_config: Optional[Dict[str, List[str]]] = None,
                  preprocessor_config: Optional[Dict[str, bool]] = None,
                  stanza_base_tokenizer_package: Optional[Union[str, Dict[str, str]]] = None,
                  normalizer_config: Optional[Dict[str, Any]] = None,
                  stanza_pipeline_config: Optional[Dict[str, Any]] = None,
                  spacy_pipeline_config: Optional[Dict[str, Any]] = None,
                  custom_pipes_config: Optional[List[Tuple[str, Dict[str, Any]]]] = None) -> Language:
    """
    This function is used to get global nlp model
    :param use_gpu: run annotation on GPU
    :param lang: model language
    :param spacy_package: spacy blank nlp package name
    :param text_meta_config: None if all records are text string, otherwise set this config for json string
    :param preprocessor_config: None if don't apply preprocessor, {} if use default preprocessor config
    :param stanza_base_tokenizer_package: None if use spacy base tokenizer otherwise use stanza base tokenizer
    :param normalizer_config: None if annotation don't apply normalizer otherwise set this config for normalizer
    :param stanza_pipeline_config: None if don't use stanza pipeline, {} if use default stanza pipeline config
    :param spacy_pipeline_config: None if don't use spacy pipeline, {} if use default spacy pipeline config
    :param custom_pipes_config: None if don't apply custom pipes, otherwise [(pipe_name, pipe_config), ...}]
    :return: global spacy nlp model
    """

    global NLP_MODEL

    if use_gpu:
        spacy.prefer_gpu()

    if NLP_MODEL is None:

        if not stanza_base_tokenizer_package and stanza_pipeline_config is not None:
            warnings.warn("Spacy base tokenizer doesn't do sentence segmentation but stanza pipeline requires input doc"
                          " has annotation of sentences, so sentence_detector will be used to do sentence detection.",
                          stacklevel=2)

        # create blank nlp
        spacy_package = spacy_package or DEFAULT_SPACY_PACKAGE
        nlp = load_blank_nlp(lang, spacy_package)

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

        # add stanza or/and spacy pipline (stanza pipeline need to run before spacy pipeline if both pipelines added)
        if stanza_pipeline_config is not None:
            stanza_pipeline_config["use_gpu"] = use_gpu
            nlp.add_pipe("stanza_pipeline", config=stanza_pipeline_config)
        if spacy_pipeline_config is not None:
            spacy_pipeline_config["package"] = spacy_package
            nlp.add_pipe("spacy_pipeline", config=spacy_pipeline_config)

        # add custom pipes
        if custom_pipes_config:
            for pipe_name, pipe_config in custom_pipes_config:
                nlp.add_pipe(pipe_name, config=pipe_config)

        logger.info(f"nlp model config (use_gpu = {use_gpu}):\n{get_nlp_model_config_str(nlp)}")

        NLP_MODEL = nlp

    return NLP_MODEL


def get_nlp_model_config_str(nlp: Language) -> str:
    table = []
    lang = nlp.lang
    spacy_package = f"{nlp.meta['name']} ({nlp.meta['version']})"
    meta_tokenizer = nlp.tokenizer
    preprocessor = meta_tokenizer.preprocessor
    base_tokenizer = meta_tokenizer.base_tokenizer
    normalizer = meta_tokenizer.normalizer
    pipe_names = nlp.pipe_names

    table.append(["lang", lang])
    table.append(["spacy_package", spacy_package])

    meta_tokenizer_config = []
    for attr in ["text_fields_in_json", "meta_fields_to_keep", "meta_fields_to_drop"]:
        attr_value = getattr(meta_tokenizer, attr)
        if attr_value:
            meta_tokenizer_config.append(f"{attr} ({', '.join(attr_value)})")
    table.append(["meta_tokenizer", ", ".join(meta_tokenizer_config)])

    table.append(["preprocessor", f"Yes ({preprocessor.get_preprocessor_config()})"] if preprocessor else "No")
    table.append(["base_tokenizer", base_tokenizer.__class__.__name__])
    if isinstance(base_tokenizer, StanzaBaseTokenizer):
        table[-1][-1] += \
            f" ({get_stanza_load_list(base_tokenizer.lang, base_tokenizer.tokenize_package, 'tokenize')[0][1]})"
    table.append(["normalizer", f"Yes ({normalizer.get_normalizer_config()})"] if normalizer else "No")

    custom_pipes = []
    for pipe_name in pipe_names:
        pipe = nlp.get_pipe(pipe_name)
        if pipe_name == "stanza_pipeline":
            stanza_load_list = get_stanza_load_list(pipe.lang, pipe.package, pipe.processors)
            stanza_load_list = ", ".join([f"{processor} ({pakcage})" for processor, pakcage in stanza_load_list])
            table.append(["stanza_pipeline", stanza_load_list])
        elif pipe_name == "spacy_pipeline":
            table.append(["spacy_pipeline", ", ".join(pipe.nlp.pipe_names)])
        else:
            custom_pipes.append(pipe_name)
    table.append(["custom_pipes", ", ".join(custom_pipes)])

    field_max_len, value_max_len = max(len(field) for field, _ in table), max(len(value) for _, value in table)
    field_format, value_format = f"<{field_max_len}", f"<{value_max_len}"
    line_row = "=" * (7 + field_max_len + value_max_len)
    table_rows = [line_row]
    table_rows.extend([f"| {field:{field_format}} | {value:{value_format}} |" for field, value in table])
    table_rows.append(line_row)
    return "\n".join(table_rows)
