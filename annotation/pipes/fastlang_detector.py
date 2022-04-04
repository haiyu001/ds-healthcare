from typing import Tuple
from fasttext.FastText import _FastText
from utils.resource_util import get_model_filepath
from spacy.tokens import Doc
import fasttext


class FastLangDetector(object):

    def __init__(self, attrs: Tuple[str, str] = ("language", "language_score"), model_name: str = "id.176.ftz"):
        self._language, self._language_score = attrs
        self.model = self.load_language_model(model_name)
        Doc.set_extension(self._language, default=None, force=True)
        Doc.set_extension(self._language_score, default=0., force=True)

    def __call__(self, doc: Doc) -> Doc:
        if doc.text.strip():
            labels, confidences = self.model.predict(doc.text)
            language, language_score = labels[0][9:], round(confidences[0], 4)
            doc._.set(self._language, language)
            doc._.set(self._language_score, language_score)
        return doc

    @staticmethod
    def load_language_model(model_name: str = "lid.176.ftz") -> _FastText:
        language_model_path = get_model_filepath("models", model_name)
        language_model = fasttext.load_model(language_model_path)
        return language_model
