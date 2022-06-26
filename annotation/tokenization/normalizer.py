from typing import Dict, List, Any, Optional, Tuple
from utils.general_util import load_json_file
from spacy.matcher import Matcher
from spacy.tokens import Doc, Token
from spacy.util import filter_spans
from spacy import Language


class Normalizer(object):

    def __init__(self,
                 nlp: Language,
                 replace_norm: Optional[Dict[str, Dict[str, Any]]] = None,
                 merge_norm: Optional[Dict[str, Dict[str, Any]]] = None,
                 split_norm: Optional[Dict[str, Dict[str, Any]]] = None,
                 canonicalization_filepath: Optional[str] = None):
        self.vocab = nlp.vocab
        self.canonicalization_filepath = canonicalization_filepath
        if self.canonicalization_filepath is None:
            self.replace_norm = replace_norm
            self.merge_norm = merge_norm
            self.split_norm = split_norm
        else:
            self.replace_norm, self.merge_norm, self.split_norm = self.load_normalization()
        if self.merge_norm:
            self.merge_matcher = self._create_merge_matcher()
        Token.set_extension("norm_text", default=None, force=True)

    def normalize(self, doc: Doc) -> Doc:
        if not self.replace_norm and not self.merge_norm and not self.split_norm:
            return doc
        norm_spaces = None
        if self.replace_norm:
            self.normalize_replace(doc)
        if self.merge_norm:
            self.normalize_merge(doc)
        if self.split_norm:
            norm_spaces = self.normalize_split(doc)
        norm_doc = self.create_norm_doc(doc, norm_spaces)
        return norm_doc

    def load_normalization(self) -> \
            Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        normalization_dict = load_json_file(self.canonicalization_filepath)
        self.replace_norm, self.merge_norm, self.split_norm = {}, {}, {}
        for id, id_normalization_dict in normalization_dict.items():
            normalization_type = id_normalization_dict["type"]
            if normalization_type == "replace":
                self.replace_norm[id] = id_normalization_dict
            elif normalization_type == "merge":
                self.merge_norm[id] = id_normalization_dict
            elif normalization_type == "split":
                self.split_norm[id] = id_normalization_dict
            else:
                raise ValueError(f"Unsupported normalization type of {normalization_type}")
        return self.replace_norm, self.merge_norm, self.split_norm

    def _create_merge_matcher(self) -> Matcher:
        matcher = Matcher(self.vocab)
        for key_id in self.merge_norm:
            case = "LOWER" if self.merge_norm[key_id]["case_insensitive"] else "TEXT"
            pattern = [{case: word} for word in self.merge_norm[key_id]["key"].split()]
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
            original_split_words.append(original_split_word)
            start_id += len(split_word)
        return original_split_words

    def normalize_replace(self, doc: Doc):
        for token in doc:
            token_text = token.text
            if token.lower_ in self.replace_norm and (self.replace_norm[token.lower_]["case_insensitive"] or
                                                      self.replace_norm[token.lower_]["key"] == token_text):
                norm_text = self._match_case(token_text, self.replace_norm[token.lower_]["value"])
                token._.set("norm_text", norm_text)

    def normalize_merge(self, doc: Doc):
        merge_spans = {doc[token_start: token_end]: self.merge_norm[self.vocab.strings[match_id]]["value"]
                       for match_id, token_start, token_end in self.merge_matcher(doc)}
        filtered_merge_spans = filter_spans(list(merge_spans.keys()))
        merge_spans = {k: v for k, v in merge_spans.items() if k in filtered_merge_spans}
        with doc.retokenize() as retokenizer:
            for merge_span in merge_spans:
                norm_text = self._match_case(merge_span.text, merge_spans[merge_span])
                attrs = {"_": {"norm_text": norm_text}}
                retokenizer.merge(merge_span, attrs=attrs)

    def normalize_split(self, doc: Doc) -> List[bool]:
        norm_spaces = []
        with doc.retokenize() as retokenizer:
            for token in doc:
                token_text = token.text
                if token.lower_ in self.split_norm and (self.split_norm[token.lower_]["case_insensitive"] or
                                                        self.split_norm[token.lower_]["key"] == token_text):
                    split_parts = self.split_norm[token.lower_]["value"].split()
                    orths = self._map_split_text(token_text, split_parts)
                    len_parts = len(split_parts)
                    dummy_heads = [(token, i) for i in range(len_parts)]
                    retokenizer.split(token, orths, heads=dummy_heads)
                    norm_spaces.extend([True] * (len_parts - 1) + [token.whitespace_.isspace()])
                else:
                    norm_spaces.append(token.whitespace_.isspace())
        return norm_spaces

    def create_norm_doc(self, doc: Doc, norm_spaces: Optional[List[bool]] = None) -> Doc:
        add_norm_space = norm_spaces is None
        add_sent_start = doc.has_annotation("SENT_START")
        sent_starts = None
        if add_norm_space:
            norm_spaces = []
        if add_sent_start:
            sent_starts = []
        norm_words = []
        org_texts_with_ws = []

        for token in doc:
            org_texts_with_ws.append(token.text + token.whitespace_)
            norm_words.append(token._.get("norm_text") or token.text)
            if add_norm_space:
                norm_spaces.append(token.whitespace_.isspace())
            if add_sent_start:
                sent_starts.append(token.is_sent_start)
        norm_doc = Doc(self.vocab,
                       words=norm_words,
                       spaces=norm_spaces,
                       sent_starts=sent_starts,
                       user_data={"org_texts_with_ws": org_texts_with_ws})
        return norm_doc

    def get_normalizer_config(self) -> str:
        return f" ({self.canonicalization_filepath})" if self.canonicalization_filepath is not None else ""
