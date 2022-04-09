from typing import Union, Dict
from pathlib import Path
from subprocess import call
import stanza
import os

MODELS_HOME = os.environ["MODELS_HOME"]


def get_model_filepath(*parts: str) -> str:
    path = MODELS_HOME
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
    repo_ignore = ["-x", f"{repo_name}/log/*", f"{repo_name}/test/*", f"{repo_name}/tmp/*", f"{repo_name}/.*"]
    call(zip_command + repo_ignore)
    os.chdir(cwd)
    return repo_zip_filepath


def get_stanza_model_dir() -> str:
    model_dir = get_model_filepath("stanza")
    return model_dir


def get_spacy_model_path(lang: str, package: str) -> str:
    model_path = get_model_filepath("spacy", lang, package, package.rsplit("-", 1)[0], package)
    return model_path


def download_stanza_model(lang: str, package: str = "default", processors: Union[str, Dict[str, str]] = {}):
    stanza.download(lang, get_model_filepath("stanza"), package, processors)