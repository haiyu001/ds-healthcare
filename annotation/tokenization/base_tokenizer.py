from typing import List, Tuple
from utils.resource_util import get_stanza_model_dir
from abc import ABC, abstractmethod
from spacy import Language
from spacy.tokens import Doc
import stanza
import warnings


class BaseTokenizer(ABC):

    @abstractmethod
    def tokenize(self, text: str) -> Doc:
        pass


class SpacyBaseTokenizer(BaseTokenizer):

    def __init__(self, nlp: Language):
        self.tokenizer = nlp.tokenizer

    def tokenize(self, text: str) -> Doc:
        doc = self.tokenizer(text)
        return doc

    def pipe(self, texts):
        for text in texts:
            yield self.tokenize(text)


class StanzaBaseTokenizer(BaseTokenizer):

    def __init__(self,
                 nlp: Language,
                 lang: str = "en",
                 tokenize_package: str = "default"):

        self.lang = lang
        self.tokenize_package = tokenize_package
        self.snlp = stanza.Pipeline(lang=self.lang,
                                    dir=get_stanza_model_dir(),
                                    package=None,
                                    processors={"tokenize": tokenize_package},
                                    verbose=False)
        self.vocab = nlp.vocab

    def tokenize(self, text: str) -> Doc:
        if not text:
            return Doc(self.vocab)
        elif text.isspace():
            return Doc(self.vocab, words=[text], spaces=[False])

        snlp_doc = self.snlp(text)

        text = snlp_doc.text
        snlp_tokens, sent_starts = [], []
        for sentence in snlp_doc.sentences:
            for i, token in enumerate(sentence.tokens):
                for j, word in enumerate(token.words):
                    snlp_tokens.append(word)
                    sent_starts.append(i == 0 and j == 0)
        token_texts = [token.text for token in snlp_tokens]

        try:
            words, spaces, sent_starts = self.get_words_and_spaces(token_texts, sent_starts, text)
        except ValueError:
            words = token_texts
            spaces = [True] * len(words)
            warnings.warn("Due to multiword token expansion or an alignment issue, "
                          "the original text has been replaced by space-separated expanded tokens.",
                          stacklevel=4)
        doc = Doc(self.vocab, words=words, spaces=spaces, sent_starts=sent_starts)
        return doc

    def get_words_and_spaces(self, token_texts: List[str], sent_starts: List[bool], text: str) \
            -> Tuple[List[str], List[bool], List[bool]]:
        if "".join("".join(token_texts).split()) != "".join(text.split()):
            raise ValueError("Unable to align mismatched text and words.")

        token_texts_and_sent_starts = [[t, s] for t, s in zip(token_texts, sent_starts)]
        text_words, text_spaces, text_sents = [], [], []
        text_pos = 0

        norm_words = []
        for i in range(len(token_texts_and_sent_starts)):
            if not token_texts_and_sent_starts[i][0].isspace():
                norm_words.append(token_texts_and_sent_starts[i])
            else:
                if norm_words[i][1] and i + 1 < len(token_texts_and_sent_starts):
                    token_texts_and_sent_starts[i + 1][1] = True

        for word, is_sent_start in norm_words:
            try:
                word_start = text[text_pos:].index(word)
            except ValueError:
                raise ValueError("Unable to align mismatched text and words.")
            if word_start > 0:
                text_words.append(text[text_pos: text_pos + word_start])
                text_spaces.append(False)
                text_sents.append(False)
                text_pos += word_start
            text_words.append(word)
            text_spaces.append(False)
            text_sents.append(is_sent_start)
            text_pos += len(word)
            if text_pos < len(text) and text[text_pos] == " ":
                text_spaces[-1] = True
                text_pos += 1
        if text_pos < len(text):
            text_words.append(text[text_pos:])
            text_spaces.append(False)
            text_sents.append(False)
        return text_words, text_spaces, text_sents

    def pipe(self, texts):
        for text in texts:
            yield self.tokenize(text)
