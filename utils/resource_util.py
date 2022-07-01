from typing import List
from utils.general_util import load_json_file
import os

MODELS_HOME = os.environ["MODELS_HOME"]

DATA_HOME = os.environ["DATA_HOME"]


def get_model_filepath(*parts: str) -> str:
    path = MODELS_HOME
    for part in parts:
        path = os.path.join(path, part)
    return path


def get_data_filepath(*parts: str) -> str:
    path = DATA_HOME
    for part in parts:
        path = os.path.join(path, part)
    return path


def get_spacy_model_path(lang: str, package: str) -> str:
    model_path = get_model_filepath("spacy", lang, package, package.rsplit("-", 1)[0], package)
    return model_path


def get_stanza_model_dir() -> str:
    model_dir = get_model_filepath("stanza")
    return model_dir


def load_nltk_stop_words(lang: str) -> List[str]:
    stop_words_dir = get_model_filepath("NLTK", "corpora", "stopwords")
    language_filepath = {lang: os.path.join(stop_words_dir, lang) for lang in os.listdir(stop_words_dir)}
    if lang not in language_filepath:
        raise Exception(f"NLTK stop words does not exist for language {lang}")
    else:
        stop_words = []
        with open(language_filepath[lang], "r") as input:
            for line in input:
                stop_words.append(line.strip())
        return stop_words


def load_stop_words(filter_min_count: int = 15) -> List[str]:
    stop_words_filepath = get_model_filepath("lexicon", "stop_words.json")
    stop_words = load_json_file(stop_words_filepath)
    stop_words = [k for k, v in stop_words.items() if v >= filter_min_count]
    return stop_words


if __name__ == "__main__":
    import stanza

    lang = "en"
    package = "mimic"
    processors = {"ner": "i2b2"}
    stanza.download(lang, get_model_filepath("stanza"), package, processors)

    # nltk.download("stopwords")
