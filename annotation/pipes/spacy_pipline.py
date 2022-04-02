from annotation.annotation_utils.annotate_util import get_spacy_model_path
from annotation.pipes.factories import *
from spacy.tokens import Doc
import spacy


class SpacyPipeline:

    def __init__(self,
                 lang: str = "en",
                 package: str = "en_core_web_md-3.2.0",
                 exclude: List[str] = [],
                 sentence_detector: bool = False):

        spacy_model_path = get_spacy_model_path(lang, package)
        self.nlp = spacy.load(spacy_model_path, exclude=exclude)
        pipe_names = set(self.nlp.pipe_names)
        if sentence_detector:
            self.nlp.add_pipe("sentence_detector", before="parser" if "parser" in pipe_names else None)

    def __call__(self, doc: Doc) -> Doc:
        doc = self.nlp(doc)
        return doc
