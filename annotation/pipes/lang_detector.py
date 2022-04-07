from typing import Tuple
from langdetect.lang_detect_exception import LangDetectException
from langdetect import detect_langs
from spacy.tokens import Doc


class LangDetector(object):
    """
    This language detector is for multiprocessing
    """

    def __init__(self, attrs: Tuple[str, str] = ("language", "language_score")):
        self._language, self._language_score = attrs
        Doc.set_extension(self._language, default=None, force=True)
        Doc.set_extension(self._language_score, default=0., force=True)

    def __call__(self, doc: Doc) -> Doc:
        if doc.text.strip():
            try:
                detected_language = detect_langs(doc.text)[0]
                language, language_score = str(detected_language.lang), round(float(detected_language.prob), 4)
            except LangDetectException:
                language, language_score = "UNKNOWN", 0.0
            doc._.set(self._language, language)
            doc._.set(self._language_score, language_score)
        return doc
