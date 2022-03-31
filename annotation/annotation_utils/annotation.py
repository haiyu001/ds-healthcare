from typing import Dict, Any, Union, List
from utils.resources import get_model_file_path
from spacy.tokens import Doc
from spacy import Language
import stanza
import spacy


def get_stanza_model_dir() -> str:
    model_dir = get_model_file_path("stanza")
    return model_dir


def get_spacy_model_path(lang: str, package: str) -> str:
    model_path = get_model_file_path("spacy", lang, package, package.rsplit("-", 1)[0], package)
    return model_path


def download_stanza_model(lang: str, package: str = "default", processors: Union[str, Dict[str, str]] = {}):
    stanza.download(lang, get_model_file_path("stanza"), package, processors)


def load_blank_nlp(lang: str, package: str, exclude: List[str] =
                   ["tok2vec", "tagger", "parser", "ner", "attribute_ruler", "lemmatizer"]) -> Language:
    spacy_model_path = get_spacy_model_path(lang, package)
    blank_nlp = spacy.load(spacy_model_path, exclude=exclude)
    return blank_nlp


def doc_to_dict(doc: Doc) -> Dict[str, Any]:

    data = {"text": doc.text, "tokens": [], "_": {}}
    
    # tokenization info (metadata, source_text)
    for extension in ["metadata", "source_text"]:
        if doc.has_extension(extension):
            data["_"][extension] = doc._.get(extension)
    
    # spacy/stanza pipline (sentence, NER and tokens)
    if doc.has_annotation("SENT_START"):
        data["sentences"] = [{"start_id": sent.start, "end_id": sent.end} for sent in doc.sents]

    if doc.has_annotation("ENT_IOB"):
        data["ents"] = [{"start_id": ent.start,
                         "end_id": ent.end,
                         "entity": ent.label_,
                         "text": ent.text} 
                        for ent in doc.ents]

    for token in doc:
        token_data = {"id": token.i,
                      "start_char": token.idx,
                      "end_char": token.idx + len(token),
                      "text": token.text,
                      "whitespace": token.whitespace_}
        if doc.has_annotation("LEMMA"):
            token_data["lemma"] = token.lemma_
        if doc.has_annotation("TAG"):
            token_data["pos"] = token.pos_
            token_data["tag"] = token.tag_
        if doc.has_annotation("DEP"):
            token_data["gov"] = token.head.i
            token_data["rel"] = token.dep_
        data["tokens"].append(token_data)

    # custom pipes
    if doc.has_extension("language"):
        data["_"]["language"] = {"lang": doc._.get("language"), "lang_score": doc._.get("language_score")}

    if doc.has_extension("phrases"):
        data["_"]["phrases"] = doc._.get("phrases")

    return data
