from typing import Optional, Dict, List, Iterator
from spacy.lang.lex_attrs import like_email, like_url, is_punct, like_num
from annotation.annotation_utils.annotator_util import load_blank_nlp
from pyspark.sql import Column
from pyspark.sql.types import StringType
import pyspark.sql.functions as F
from spacy import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc
from spacy.util import filter_spans
import pandas as pd


def pudf_get_corpus_line(text_iter: Column,
                         lang: str,
                         spacy_package: str,
                         ngram_match_dict: Optional[Dict[str, str]] = None) -> Column:
    def get_corpus_line(text_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
        nlp = load_blank_nlp(lang, spacy_package, whitespace_tokenizer=True)
        ngram_matcher = None
        if ngram_match_dict is not None:
            ngram_matcher = get_ngram_matcher(nlp, list(ngram_match_dict.keys()))
        for text in text_iter:
            doc = text.apply(nlp)
            if ngram_matcher is not None:
                doc = doc.apply(match_ngram, ngram_matcher=ngram_matcher)
            corpus_line = doc.apply(doc_to_corpus_line, ngram_match_dict=ngram_match_dict)
            yield corpus_line

    return F.pandas_udf(get_corpus_line, StringType())(text_iter)


def is_valid_token(text: str) -> bool:
    return len(text) > 0 and not like_email(text) and not like_url(text) and not like_num(text) and not is_punct(text)


def doc_to_corpus_line(doc: Doc, ngram_match_dict: Optional[Dict[str, str]] = None) -> str:
    token_texts = []
    for token in doc:
        token_text = token.text
        if ngram_match_dict is not None:
            token_text = ngram_match_dict.get(token_text, token_text)
        if token_text:
            token_texts.append(token_text)
    corpus_line = " ".join(token_texts)
    return corpus_line


def match_ngram(doc: Doc, ngram_matcher):
    matches = ngram_matcher(doc)
    spans = []
    for _, start, end in matches:
        span = doc[start: end]
        spans.append(span)
    with doc.retokenize() as retokenizer:
        for span in filter_spans(spans):
            retokenizer.merge(span)
    return doc


def get_ngram_matcher(nlp: Language, ngrams: List[str]):
    ngram_matcher = PhraseMatcher(nlp.vocab)
    ngrams_docs = list(nlp.tokenizer.pipe(ngrams))
    ngram_matcher.add("ngram_match", ngrams_docs)
    return ngram_matcher


