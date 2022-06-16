from pathlib import Path
from subprocess import call
import os
from typing import List

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


def get_repo_dir() -> str:
    return str(Path(__file__).parent.parent)


def zip_repo(repo_zip_dir: str) -> str:
    cwd = os.getcwd()
    repo_dir = get_repo_dir()
    repo_name = Path(repo_dir).stem
    os.chdir(Path(repo_dir).parent)
    repo_zip_filepath = os.path.join(repo_zip_dir, f"{repo_name}.zip")
    zip_command = ["zip", "-r", repo_zip_filepath, repo_name]
    repo_ignore = ["-x",
                   f"{repo_name}/logs/*",
                   f"{repo_name}/test/*",
                   f"{repo_name}/tmp/*",
                   f"{repo_name}/notebooks/*",
                   f"{repo_name}/.*"]
    call(zip_command + repo_ignore)
    os.chdir(cwd)
    return repo_zip_filepath


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


if __name__ == "__main__":
    import stanza
    import nltk

    lang = "en"
    package = "mimic"
    processors = {"ner": "i2b2"}
    stanza.download(lang, get_model_filepath("stanza"), package, processors)

    nltk.download("stopwords")
