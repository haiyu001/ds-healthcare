from typing import Dict, Union, List, Any, Optional, Tuple
from pyspark.sql import SparkSession, DataFrame
from utils.config_util import read_config, config_type_casting, clean_config_str
from utils.general_util import get_filepaths_recursively
from utils.resource_util import get_stanza_model_dir, get_spacy_model_path
from stanza.resources.common import process_pipeline_parameters, maintain_processor_list
from spacy.tokenizer import Tokenizer
from spacy import Language
import spacy
import json
import os
import re

DEFAULT_SPACY_PACKAGE = "en_core_web_md-3.2.0"


def get_spacy_model_pipes(spacy_model_path: str) -> List[str]:
    spacy_meta_filepath = os.path.join(spacy_model_path, "meta.json")
    with open(spacy_meta_filepath) as infile:
        spacy_meta = json.load(infile)
    pipes = spacy_meta["pipeline"]
    return pipes


def load_blank_nlp(lang: str, package: str, whitespace_tokenizer: bool = False) -> Language:
    spacy_model_path = get_spacy_model_path(lang, package)
    spacy_model_pipes = get_spacy_model_pipes(spacy_model_path)
    blank_nlp = spacy.load(spacy_model_path, exclude=spacy_model_pipes)
    if whitespace_tokenizer:
        blank_nlp.tokenizer = Tokenizer(blank_nlp.vocab, token_match=re.compile(r'\S+').match)
    return blank_nlp


def get_stanza_load_list(lang: str = "en",
                         package: str = "default",
                         processors: Union[str, Dict[str, str]] = {}) -> List[List[str]]:
    stanza_dir = get_stanza_model_dir()
    resources_filepath = os.path.join(stanza_dir, "resources.json")
    with open(resources_filepath) as infile:
        resources = json.load(infile)
    lang, _, package, processors = process_pipeline_parameters(lang, stanza_dir, package, processors)
    stanza_load_list = maintain_processor_list(resources, lang, package, processors)
    return stanza_load_list


def read_nlp_model_config(config_filepath: str) -> Dict[str, Any]:
    config = read_config(config_filepath)

    optional_sections = ["Preprocessor", "Normalizer", "StanzaPipeline", "SpacyPipeline"]
    optional_section_configs = {}
    for section in optional_sections:
        section_config = config_type_casting(config.items(section))
        if not section_config.pop(section):
            section_config = None
        optional_section_configs[section] = section_config

    custom_pipes_params = {}
    for section in config.sections():
        if section.startswith("CustomPipes:"):
            custom_pipe_name = section.split(":")[-1].strip()
            custom_pipes_params[custom_pipe_name] = config_type_casting(config.items(section))

    nlp_model_config = dict(
        use_gpu=config["Annotator"].getboolean("use_gpu"),
        lang=clean_config_str(config["Annotator"]["lang"]),
        spacy_package=clean_config_str(config["Annotator"]["spacy_package"]),
        meta_tokenizer_config=config_type_casting(config.items("MetaTokenizer")),
        preprocessor_config=optional_section_configs["Preprocessor"],
        stanza_base_tokenizer_package=clean_config_str(config["BaseTokenizer"]["stanza_base_tokenizer_package"]),
        normalizer_config=optional_section_configs["Normalizer"],
        stanza_pipeline_config=optional_section_configs["StanzaPipeline"],
        spacy_pipeline_config=optional_section_configs["SpacyPipeline"],
        custom_pipes_config=[(k, custom_pipes_params.get(k, {})) for k, v in
                             config_type_casting(config.items("CustomPipes")).items() if v])
    return nlp_model_config


def read_annotation_config(config_filepath: str) -> Dict[str, Any]:
    config = read_config(config_filepath)
    annotation_config = {}
    for section in config.sections():
        annotation_config.update(config_type_casting(config.items(section)))
    return annotation_config


def load_annotation(spark: SparkSession,
                    annotation_dir: str,
                    drop_non_english: bool = True,
                    num_partitions: Optional[int] = None) -> DataFrame:
    annotation_filepaths = get_filepaths_recursively(annotation_dir, ["json", "txt"])
    annotation_sdf = spark.read.json(annotation_filepaths)
    if drop_non_english:
        annotation_sdf = annotation_sdf.filter(annotation_sdf["_"]["language"]["lang"] == "en")
    if num_partitions is not None:
        annotation_sdf = annotation_sdf.repartion(num_partitions)
    return annotation_sdf
