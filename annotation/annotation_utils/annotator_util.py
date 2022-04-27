from typing import Dict, List, Any
from annotation.pipes.stanza_pipeline import get_stanza_processors
from utils.config_util import read_config, config_type_casting, clean_config_str
from utils.resource_util import get_spacy_model_path
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


def read_annotation_config(config_filepath: str) -> Dict[str, Any]:
    config = read_config(config_filepath)
    annotation_config = {}
    for section in config.sections():
        annotation_config.update(config_type_casting(config.items(section)))
    return annotation_config


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

    custom_pipes_config = {}
    for custom_pipe_name, add_custom_pipe in config_type_casting(config.items("CustomPipes")).items():
        if add_custom_pipe:
            custom_pipes_config[custom_pipe_name] = custom_pipes_params.get(custom_pipe_name, {})

    nlp_model_config = dict(
        use_gpu=config["Annotator"].getboolean("use_gpu"),
        lang=clean_config_str(config["Annotator"]["lang"]),
        spacy_package=clean_config_str(config["Annotator"]["spacy_package"]),
        metadata_tokenizer_config=config_type_casting(config.items("MetadataTokenizer")),
        preprocessor_config=optional_section_configs["Preprocessor"],
        stanza_base_tokenizer_package=clean_config_str(config["BaseTokenizer"]["stanza_base_tokenizer_package"]),
        normalizer_config=optional_section_configs["Normalizer"],
        stanza_pipeline_config=optional_section_configs["StanzaPipeline"],
        spacy_pipeline_config=optional_section_configs["SpacyPipeline"],
        custom_pipes_config=None if not custom_pipes_config else custom_pipes_config,)
    return nlp_model_config


def get_canonicalization_nlp_model_config(nlp_model_config_filepath: str) -> Dict[str, Any]:
    nlp_model_config = read_nlp_model_config(nlp_model_config_filepath)
    nlp_model_config["metadata_tokenizer_config"]["ignore_metadata"] = True
    nlp_model_config["normalizer_config"] = None
    # set language_detector and spell_detector
    if "fastlang_detector" in nlp_model_config["custom_pipes_config"]:
        nlp_model_config["custom_pipes_config"] = {"fastlang_detector": {}}
    else:
        nlp_model_config["custom_pipes_config"] = {"lang_detector": {}}
    nlp_model_config["custom_pipes_config"].update({"spell_detector": {}})

    # set pos detector and lemma detector
    if nlp_model_config["stanza_pipeline_config"]:
        stanza_processors = get_stanza_processors(nlp_model_config["stanza_pipeline_config"]["processors"],
                                                  nlp_model_config["stanza_pipeline_config"]["processors_packages"])
        stanza_pos_lemma = ["tokenize", "pos", "lemma"]
        if not stanza_processors:
            nlp_model_config["stanza_pipeline_config"]["processors"] = ",".join(stanza_pos_lemma)
        elif isinstance(stanza_processors, str):
            nlp_model_config["stanza_pipeline_config"]["processors"] = \
                ",".join([i for i in stanza_pos_lemma if i in stanza_processors])
        else:
            pos_lemma_processors, pos_lemma_processors_packages = [], []
            for i in stanza_pos_lemma:
                if i in stanza_processors:
                    pos_lemma_processors.append(i)
                    pos_lemma_processors_packages.append(stanza_processors[i])
            nlp_model_config["stanza_pipeline_config"]["processors"] = ",".join(pos_lemma_processors)
            nlp_model_config["stanza_pipeline_config"]["processors_packages"] = ",".join(pos_lemma_processors_packages)

        processors = nlp_model_config["stanza_pipeline_config"]["processors"]
        if "pos" in processors and "lemma" in processors:
            nlp_model_config["spacy_pipeline_config"] = None
        if "pos" not in processors and "lemma" not in processors:
            nlp_model_config["stanza_pipeline_config"] = None
            return nlp_model_config

    if nlp_model_config["spacy_pipeline_config"]:
        nlp_model_config["spacy_pipeline_config"]["sentence_detector"] = False
        pipes = nlp_model_config["spacy_pipeline_config"]["pipes"]
        spacy_pos_lemma = ["tok2vec", "tagger", "attribute_ruler", "lemmatizer"]
        if not pipes:
            nlp_model_config["spacy_pipeline_config"]["pipes"] = ",".join(spacy_pos_lemma)
        else:
            pos_lemma_pipes = []
            for i in spacy_pos_lemma:
                if i in pipes:
                    pos_lemma_pipes.append(i)
            nlp_model_config["spacy_pipeline_config"]["pipes"] = ",".join(pos_lemma_pipes)
    return nlp_model_config


def get_nlp_model_config(nlp_model_config_filepath: str, normalization_json_filepath: str) -> Dict[str, Any]:
    nlp_model_config = read_nlp_model_config(nlp_model_config_filepath)
    if nlp_model_config["normalizer_config"] is not None:
        nlp_model_config["normalizer_config"].update({"normalization_json_filepath": normalization_json_filepath})
    return nlp_model_config


