from typing import Any, Iterator
from annotation.tokenization.base_tokenizer import SpacyBaseTokenizer, StanzaBaseTokenizer
from annotation.annotation_utils.annotation_util import load_blank_nlp, DEFAULT_SPACY_PACKAGE, get_stanza_load_list
from annotation.tokenization.normalizer import Normalizer
from annotation.tokenization.preprocessor import Preprocessor
from annotation.tokenization.tokenizer import MetaTokenizer
from utils.log_util import get_logger
from annotation.pipes.factories import *
from spacy.tokens import Doc
from pyspark.sql.types import StringType
from pyspark.sql import Column, functions as F
from threading import Lock
import pandas as pd
import json
import warnings
import spacy


class SingletonMeta(type):
    _instances = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class Annotator(metaclass=SingletonMeta):
    nlp: Language = None

    def __init__(self,
                 use_gpu: bool = False,
                 lang: str = "en",
                 spacy_package: Optional[str] = None,
                 meta_tokenizer_config: Optional[Dict[str, List[str]]] = None,
                 preprocessor_config: Optional[Dict[str, bool]] = None,
                 stanza_base_tokenizer_package: Optional[Union[str, Dict[str, str]]] = None,
                 normalizer_config: Optional[Dict[str, Any]] = None,
                 stanza_pipeline_config: Optional[Dict[str, Any]] = None,
                 spacy_pipeline_config: Optional[Dict[str, Any]] = None,
                 custom_pipes_config: Optional[List[Tuple[str, Dict[str, Any]]]] = None):
        """
        :param use_gpu: run annotation on GPU
        :param lang: model language
        :param spacy_package: spacy blank nlp package name
        :param meta_tokenizer_config: None if all records are text string, otherwise set this config for json string
        :param preprocessor_config: None if don't apply preprocessor, {} if use default preprocessor config
        :param stanza_base_tokenizer_package: None if use spacy base tokenizer otherwise use stanza base tokenizer
        :param normalizer_config: None if annotation don't apply normalizer otherwise set this config for normalizer
        :param stanza_pipeline_config: None if don't use stanza pipeline, {} if use default stanza pipeline config
        :param spacy_pipeline_config: None if don't use spacy pipeline, {} if use default spacy pipeline config
        :param custom_pipes_config: None if don't apply custom pipes, otherwise [(pipe_name, pipe_config), ...}]
        :return: global spacy nlp model
        """

        if use_gpu:
            spacy.prefer_gpu()

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

        nlp.tokenizer = MetaTokenizer(base_tokenizer, preprocessor, normalizer, **meta_tokenizer_config) \
            if meta_tokenizer_config is not None else MetaTokenizer(base_tokenizer, preprocessor, normalizer)

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

        logger = get_logger()
        logger.info(f"nlp model config (use_gpu = {use_gpu}):\n{get_nlp_model_config_str(nlp)}")

        self.nlp = nlp


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
            stanza_load_list = ", ".join([f"{processor} ({pkg})" for processor, pkg in stanza_load_list])
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


def doc_to_dict(doc: Doc) -> Dict[str, Any]:
    data = {"text": doc.text,
            "tokens": [],
            "_": {}, }

    # tokenization info (metadata, source_text)
    for extension in ["metadata", "source_text"]:
        if doc.has_extension(extension):
            data["_"][extension] = doc._.get(extension)

    # spacy/stanza pipline (sentence, NER and tokens)
    if doc.has_annotation("SENT_START"):
        data["sentences"] = [{"start_id": sent.start,
                              "end_id": sent.end, } for sent in doc.sents]

    if doc.has_annotation("ENT_IOB"):
        data["ents"] = [{"start_id": ent.start,
                         "end_id": ent.end,
                         "entity": ent.label_,
                         "text": ent.text, } for ent in doc.ents]

    for token in doc:
        token_data = {"id": token.i,
                      "start_char": token.idx,
                      "end_char": token.idx + len(token),
                      "text": token.text,
                      "whitespace": token.whitespace_, }
        if doc.has_annotation("LEMMA"):
            token_data["lemma"] = token.lemma_
        if doc.has_annotation("TAG"):
            token_data["pos"] = token.pos_
            token_data["tag"] = token.tag_
        if doc.has_annotation("DEP"):
            token_data["gov"] = token.head.i
            token_data["rel"] = token.dep_
        if doc.has_annotation("MORPH"):
            token_data["morph"] = str(token.morph)
        data["tokens"].append(token_data)

    # custom pipes
    if doc.has_extension("language"):
        data["_"]["language"] = {"lang": doc._.get("language"),
                                 "score": doc._.get("language_score"), }

    if doc.has_extension("phrases"):
        data["_"]["phrases"] = doc._.get("phrases")

    if doc.has_extension("sentence_sentiments"):
        data["_"]["sentence_sentiments"] = doc._.get("sentence_sentiments")

    return data


def doc_to_json_str(doc: Doc) -> str:
    doc_dict = doc_to_dict(doc)
    doc_json_str = json.dumps(doc_dict, ensure_ascii=False)
    return doc_json_str


def pudf_annotate(text_iter: Column, nlp_model_config: Dict[str, Any]) -> Column:
    def annotate(text_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
        nlp = Annotator(**nlp_model_config).nlp
        for text in text_iter:
            doc = text.apply(nlp)
            doc_annotation_str = doc.apply(doc_to_json_str)
            yield doc_annotation_str

    return F.pandas_udf(annotate, StringType())(text_iter)
