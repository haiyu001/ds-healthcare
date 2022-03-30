from typing import Tuple
from utils.resources import get_model_file_path
from spacy.tokens import Doc
from fasttext.FastText import _FastText
import fasttext


class LangDetector(object):

    def __init__(self, attrs: Tuple[str, str] = ("language", "language_score"), model_name: str = "lid.176.ftz"):
        self._language, self._language_score = attrs
        self.model = self.get_language_model(model_name)
        Doc.set_extension(self._language, default=None, force=True)
        Doc.set_extension(self._language_score, default=0., force=True)

    def __call__(self, doc: Doc) -> Doc:
        if doc.text.strip():
            labels, confidences = self.model.predict(doc.text)
            language = labels[0].replace("__label__", "")
            language_score = round(confidences[0], 4)
            doc._.set(self._language, language)
            doc._.set(self._language_score, language_score)
        return doc

    @staticmethod
    def get_language_model(model_name: str = "lid.176.ftz") -> _FastText:
        language_model_path = get_model_file_path("models", model_name)
        language_model = fasttext.load_model(language_model_path)
        return language_model
