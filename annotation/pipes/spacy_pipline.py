from typing import Optional
from annotation.annotation_utils.annotator_util import DEFAULT_SPACY_PACKAGE, get_spacy_model_pipes
from utils.resource_util import get_spacy_model_path
from spacy.tokens import Doc
import spacy


class SpacyPipeline(object):

    def __init__(self,
                 lang: str = "en",
                 package: Optional[str] = None,
                 pipes: Optional[str] = None,
                 sentence_detector: bool = False):
        spacy_package = package or DEFAULT_SPACY_PACKAGE
        spacy_model_path = get_spacy_model_path(lang, spacy_package)
        exclude = self._get_exclude(spacy_model_path, pipes)
        self.nlp = spacy.load(spacy_model_path, exclude=exclude)
        if sentence_detector:
            self.nlp.add_pipe("sentence_detector", before="parser" if "parser" in self.nlp.pipe_names else None)

    def _get_exclude(self, spacy_model_path, pipes):
        spacy_model_pipes = get_spacy_model_pipes(spacy_model_path)
        pipes = pipes.split(",") if pipes else []
        exclude = [pipe for pipe in spacy_model_pipes if pipe not in pipes] if pipes else []
        return exclude

    def __call__(self, doc: Doc) -> Doc:
        doc = self.nlp(doc)
        return doc
