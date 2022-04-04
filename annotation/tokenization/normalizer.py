from typing import Dict, List
from spacy.matcher import Matcher
from spacy.tokens import Doc, Token
from spacy.util import filter_spans
from spacy import Language


class Normalizer:

    def __init__(self,
                 nlp: Language,
                 replace_words: Dict[str, str] = {},
                 merge_words: Dict[str, str] = {},
                 split_words: Dict[str, str] = {},
                 replace_ignore_case: bool = True,
                 merge_ignore_case: bool = True,
                 split_ignore_case: bool = True):

        self.vocab = nlp.vocab
        self.merge_words = merge_words
        self.split_words = split_words
        self.replace_words = replace_words
        self.merge_ignore_case = merge_ignore_case
        self.split_ignore_case = split_ignore_case
        self.replace_ignore_case = replace_ignore_case
        self.merge_matcher = self._create_merge_matcher()
        Token.set_extension("norm_text", default=None, force=True)
        Token.set_extension("norm_space", default=None, force=True)

    def normalize(self, doc: Doc) -> Doc:
        self.normalize_replace(doc)
        self.normalize_merge(doc)
        norm_spaces = self.normalize_split(doc)
        norm_doc = self.create_norm_doc(doc, norm_spaces)
        return norm_doc

    def _create_merge_matcher(self) -> Matcher:
        matcher = Matcher(self.vocab)
        case = "LOWER" if self.merge_ignore_case else "TEXT"
        for key_id in self.merge_words:
            pattern = [{case: word} for word in key_id.split()]
            matcher.add(key_id, [pattern])
        return matcher

    def _match_case(self, original_text: str, replace_text: str) -> str:
        if original_text.istitle():
            return replace_text.title()
        elif original_text.isupper():
            return replace_text.upper()
        else:
            return replace_text

    def _map_split_text(self, orginal_text: str, split_words: List[str]) -> List[str]:
        original_split_words = []
        start_id = 0
        for split_word in split_words:
            original_split_word = orginal_text[start_id: start_id + len(split_word)]
            original_split_words.append(self._match_case(original_split_word, split_word))
            start_id += len(split_word)
        return original_split_words

    def normalize_replace(self, doc: Doc):
        for token in doc:
            token_text = token.text
            token_match = token.lower_ if self.replace_ignore_case else token_text
            if token_match in self.replace_words:
                norm_text = self._match_case(token_text, self.replace_words[token_match])
                token._.set("norm_text", norm_text)

    def normalize_merge(self, doc: Doc):
        merge_spans = {doc[token_start: token_end]: self.merge_words[self.vocab.strings[match_id]]
                       for match_id, token_start, token_end in self.merge_matcher(doc)}
        filtered_merge_spans = filter_spans(list(merge_spans.keys()))
        merge_spans = {k: v for k, v in merge_spans.items() if k in filtered_merge_spans}
        with doc.retokenize() as retokenizer:
            for merge_span in merge_spans:
                norm_text = self._match_case(merge_span.text, merge_spans[merge_span]["merge"])
                attrs = {"_": {"norm_text": norm_text}}
                retokenizer.merge(merge_span, attrs=attrs)

    def normalize_split(self, doc: Doc) -> List[bool]:
        norm_spaces = []
        with doc.retokenize() as retokenizer:
            for token in doc:
                token_text = token.text
                token_match = token.lower_ if self.split_ignore_case else token_text
                if token_match in self.split_words:
                    split_parts = self.split_words[token_match].split()
                    len_parts = len(split_parts)
                    orths = self._map_split_text(token_text, split_parts)
                    dummy_heads = [(token, i) for i in range(len_parts)]
                    retokenizer.split(token, orths, heads=dummy_heads)
                    norm_spaces.extend([True] * (len_parts - 1) + [token.whitespace_.isspace()])
                else:
                    norm_spaces.append(token.whitespace_.isspace())
        return norm_spaces

    def create_norm_doc(self, doc: Doc, norm_spaces: List[bool]) -> Doc:
        norm_words, sent_starts = [], [] if doc.has_annotation("SENT_START") else None
        for token in doc:
            norm_words.append(token._.get("norm_text") or token.text)
            if sent_starts is not None:
                sent_starts.append(token.is_sent_start)
        norm_doc = Doc(self.vocab, words=norm_words, spaces=norm_spaces, sent_starts=sent_starts)
        return norm_doc

    def get_normalizer_config(self) -> str:
        ignore_case_config = []
        for attr in ["replace_ignore_case", "merge_ignore_case", "split_ignore_case"]:
            ignore_case_config.append(f"{attr}={getattr(self, attr)}")
        return ", ".join(ignore_case_config)
