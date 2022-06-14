from typing import List, Dict, Any, Tuple
from spacy.tokens import Doc, Token


class PhraseDetector(object):

    def __init__(self,
                 phrase_min_words_count: int = 2,
                 phrase_max_words_count: int = 6,
                 attrs: Tuple[str] = ("phrases",)):
        self._phrases, = attrs
        self.phrase_min_words_count = phrase_min_words_count
        self.phrase_max_words_count = phrase_max_words_count
        Doc.set_extension(self._phrases, getter=self.get_phrases, force=True)

    def __call__(self, doc: Doc) -> Doc:
        return doc

    def _is_not_valid_token(self, token: Token) -> bool:
        return token.is_punct or token.is_space or token.is_stop or token.like_num or \
               token.pos_ == "ADP" or token.pos_ == "DET" or token.pos_ == "CONJ"

    def get_phrases(self, doc: Doc) -> List[Dict[str, Any]]:
        phrases = []
        for phrase_span in doc.noun_chunks:
            phrase_words_count = len(phrase_span)
            if self.phrase_min_words_count <= phrase_words_count <= self.phrase_max_words_count \
                    and all([token.ent_iob == 2 or token.ent_iob == 0 for token in phrase_span]):
                if self._is_not_valid_token(phrase_span[0]):
                    phrase_span = phrase_span[1:]
                if self._is_not_valid_token(phrase_span[-1]):
                    phrase_span = phrase_span[:-1]
                if len(phrase_span) > 1:
                    phrases.append({
                        "start_id": phrase_span.start,
                        "end_id": phrase_span.end,
                        "text": phrase_span.text,
                        "phrase_poses": [token.pos_ for token in phrase_span],
                        "phrase_lemmas": [token.lemma_ for token in phrase_span],
                        "phrase_deps": [token.dep_ for token in phrase_span],
                    })
        return phrases
