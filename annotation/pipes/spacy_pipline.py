from typing import List, Optional
from annotation.annotation_utils.annotator_util import get_spacy_model_path, DEFAULT_SPACY_PACKAGE
from spacy.tokens import Doc
import spacy


class SpacyPipeline:

    def __init__(self,
                 lang: str = "en",
                 package: Optional[str] = None,
                 exclude: Optional[str] = None,
                 sentence_detector: bool = False):

        spacy_package = package or DEFAULT_SPACY_PACKAGE
        spacy_model_path = get_spacy_model_path(lang, spacy_package)
        exclude = exclude.split(',') if exclude else []

        self.nlp = spacy.load(spacy_model_path, exclude=exclude)
        pipe_names = set(self.nlp.pipe_names)
        if sentence_detector:
            self.nlp.add_pipe("sentence_detector", before="parser" if "parser" in pipe_names else None)

    def __call__(self, doc: Doc) -> Doc:
        doc = self.nlp(doc)
        return doc
