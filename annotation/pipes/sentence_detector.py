from spacy.tokens import Doc
import pysbd


class SentenceDetector(object):
    """ this component need to add before parser for spacy pipeline"""

    def __init__(self, lang: str = "en"):
        self.lang = lang

    def __call__(self, doc: Doc) -> Doc:
        seg = pysbd.Segmenter(language=self.lang, clean=False, char_span=True)
        sentences_char_spans = seg.segment(doc.text_with_ws)
        start_token_ids = set(sent.start for sent in sentences_char_spans)
        for token in doc:
            token.is_sent_start = token.idx in start_token_ids
        return doc
