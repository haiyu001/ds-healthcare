from typing import List, Dict, Any, Tuple
from pytextrank import BaseTextRank
from pytextrank.util import default_scrubber
from collections import defaultdict
from spacy.tokens import Doc, Token
import json


class PhraseDetector(object):

    def __init__(self, attrs: Tuple[str] = ("phrases",)):
        self._phrases, = attrs
        Doc.set_extension(self._phrases, getter=self.get_phrases, force=True)

    def __call__(self, doc: Doc) -> Doc:
        return doc

    def _is_valid_token(self, token: Token) -> bool:
        return not (token.is_punct or token.is_space or token.is_stop or
                   token.pos_ == 'ADP' or token.pos_ == 'DET' or token.pos_ == 'CONJ')

    def get_phrases(self, doc: Doc) -> List[Dict[str, Any]]:
        text_rank = BaseTextRank(doc,
                                 edge_weight=1.0,
                                 pos_kept=["ADJ", "NOUN", "PROPN", "VERB"],
                                 token_lookback=3,
                                 scrubber=default_scrubber,
                                 stopwords=defaultdict(list))

        spacy_phrases = {chunk.text.lower(): chunk for chunk in doc.noun_chunks}
        phrases = []
        for p in text_rank.calc_textrank():
            phrase_text = p.text.lower()
            if phrase_text in spacy_phrases and len(phrase_text.split()) > 1:
                phrase_span = spacy_phrases[phrase_text]
                # ent_iob: 2 means it is outside an entity and 0 means no entity tag is set.
                if len([token for token in phrase_span if (not token.is_stop and not token.is_punct)]) > 1 and \
                        all([token.ent_iob == 2 or token.ent_iob == 0 for token in phrase_span]):

                    if not self._is_valid_token(phrase_span[0]):
                        phrase_span = phrase_span[1:]
                    if not self._is_valid_token(phrase_span[-1]):
                        phrase_span = phrase_span[:-1]

                    phrases.append({
                        "start_id": phrase_span.start,
                        "end_id": phrase_span.end,
                        "text": phrase_span.text,
                        "phrase_rank": p.rank,
                        "phrase_count": p.count,
                        "phrase_poses": json.dumps([token.pos_ for token in phrase_span], ensure_ascii=False),
                        "phrase_lemmas": json.dumps([token.lemma_ for token in phrase_span], ensure_ascii=False),
                        "phrase_deps": json.dumps([token.dep_ for token in phrase_span], ensure_ascii=False),
                    })
        return phrases
