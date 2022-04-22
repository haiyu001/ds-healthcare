from typing import Dict, Tuple, List, Optional, Union
from stanza.resources.common import process_pipeline_parameters, maintain_processor_list
from utils.resource_util import get_stanza_model_dir
from annotation.pipes.sentence_detector import SentenceDetector
from spacy.tokens import Doc, Token
from spacy import Language
from stanza.models.common.pretrain import Pretrain
from stanza.models.common.vocab import UNK_ID
from stanza.models.common.doc import Document, Word
from stanza import Pipeline
from numpy import ndarray
import json
import os


class StanzaPipeline(object):

    def __init__(self,
                 nlp: Language,
                 lang: str = "en",
                 package: Optional[str] = "default",
                 processors: Optional[str] = None,
                 processors_packages: Optional[str] = None,
                 use_gpu: bool = False,
                 set_token_vector_hooks: bool = False,
                 attrs: Tuple[str, str, str, str] =
                 ("metadata", "source_text", "preprocessed_text", "sentence_sentiments")):

        self.lang = lang
        self.package = package
        self.processors = get_stanza_processors(processors, processors_packages)
        self.vocab = nlp.vocab
        self.use_gpu = use_gpu
        self.snlp = Pipeline(lang=self.lang,
                             dir=get_stanza_model_dir(),
                             package=self.package,
                             processors=self.processors,
                             use_gpu=self.use_gpu,
                             tokenize_pretokenized=True,
                             verbose=False)
        self.loaded_processors = {processor for processor, _ in
                                  get_stanza_load_list(self.lang, self.package, self.processors)}
        self.svecs = self._find_embeddings(self.snlp) if set_token_vector_hooks else None
        self._metadata, self._source_text, self._preprocessed_text, self._sentiment = attrs
        if "sentiment" in processors:
            Doc.set_extension(self._sentiment, default=None, force=True)

    def __call__(self, doc: Doc) -> Doc:
        spacy_doc = doc
        token_texts, token_spaces = self.convert_doc_to_tokens(spacy_doc)
        snlp_doc = self.snlp(token_texts)

        tokens, heads, sent_starts, spaces = self.get_tokens_with_heads(snlp_doc, token_spaces)
        pos, tags, morphs, lemmas, deps = [], [], [], [], []
        for token in tokens:
            pos.append(token.upos or "")
            tags.append(token.xpos or token.upos or "")
            morphs.append(token.feats or "")
            lemmas.append(token.lemma or "")
            deps.append(token.deprel or "")

        words = [t.text for t in tokens]
        heads = [head + i for i, head in enumerate(heads)]
        doc = Doc(self.vocab, words=words, spaces=spaces, sent_starts=sent_starts,
                  pos=pos if "pos" in self.loaded_processors else None,
                  tags=tags if "pos" in self.loaded_processors else None,
                  morphs=morphs if "pos" in self.loaded_processors else None,
                  lemmas=lemmas if "lemma" in self.loaded_processors else None,
                  deps=deps if "depparse" in self.loaded_processors else None,
                  heads=heads if "depparse" in self.loaded_processors else None,
                  user_data=spacy_doc.user_data)

        self.set_named_entities(doc, snlp_doc, token_texts, token_spaces)

        if self.svecs is not None:
            doc.user_token_hooks["vector"] = self.token_vector
            doc.user_token_hooks["has_vector"] = self.token_has_vector

        doc._.set(self._metadata, spacy_doc._.get(self._metadata))
        doc._.set(self._source_text, spacy_doc._.get(self._source_text))
        doc._.set(self._preprocessed_text, spacy_doc._.get(self._preprocessed_text))
        if doc.has_extension(self._sentiment):
            # 0: negative, 1: neutral, 2: positive
            doc._.set(self._sentiment, [sentence.sentiment for sentence in snlp_doc.sentences])

        return doc

    def convert_doc_to_tokens(self, doc: Doc) -> Tuple[List[List[str]], List[bool]]:
        token_texts = []
        token_spaces = []
        if not doc.has_annotation("SENT_START"):
            sentence_detector = SentenceDetector(self.lang)
            doc = sentence_detector(doc)
        for token in doc:
            if token.is_sent_start:
                token_texts.append([])
            token_texts[-1].append(token.text)
            token_spaces.append(token.whitespace_.isspace())
        return token_texts, token_spaces

    def get_tokens_with_heads(self, snlp_doc: Document, token_spaces: List[bool]) \
            -> Tuple[List[Word], List[int], List[bool], List[str]]:
        tokens, heads, sents, spaces = [], [], [], []
        offset = 0
        token_id = 0
        for sentence in snlp_doc.sentences:
            for i, token in enumerate(sentence.tokens):
                white_space, end = token_spaces[token_id], len(token.words) - 1
                for j, word in enumerate(token.words):
                    head = word.head - 1 + offset - len(tokens) if word.head else 0
                    heads.append(head)
                    tokens.append(word)
                    spaces.append("" if j < end else white_space)
                    sents.append(i == 0 and j == 0)
                token_id += 1
            offset += sum(len(token.words) for token in sentence.tokens)
        return tokens, heads, sents, spaces

    def set_named_entities(self, doc: Doc, snlp_doc: Document, token_texts: List[List[str]], token_spaces: List[bool]):
        flatten_token_texts = sum(token_texts, [])
        stanza_text = " ".join(flatten_token_texts)
        spacy_text = "".join(f"{t}{' ' * s}" for t, s in zip(flatten_token_texts, token_spaces))
        stanza_to_spacy = dict()
        spacy_id = 0
        len_stanza_text = len(stanza_text)
        for stanza_id in range(len_stanza_text):
            stanza_to_spacy[stanza_id] = spacy_id
            if stanza_text[stanza_id] == spacy_text[spacy_id]:
                spacy_id += 1
        stanza_to_spacy[len_stanza_text] = spacy_id

        ents = []
        for ent in snlp_doc.entities:
            ent_span = doc.char_span(stanza_to_spacy[ent.start_char], stanza_to_spacy[ent.end_char], ent.type)
            ents.append(ent_span)
        doc.ents = ents

    def token_vector(self, token: Token) -> ndarray:
        unit_id = self.svecs.vocab.unit2id(token.text)
        return self.svecs.emb[unit_id]

    def token_has_vector(self, token: Token) -> bool:
        return self.svecs.vocab.unit2id(token.text) != UNK_ID

    def _find_embeddings(self, snlp: Pipeline) -> Pretrain:
        embs = None
        for proc in snlp.processors.values():
            if hasattr(proc, "pretrain") and isinstance(proc.pretrain, Pretrain):
                embs = proc.pretrain
                break
        return embs


def get_stanza_processors(processors: Optional[str],
                          processors_packages: Optional[str]) -> Union[str, Dict[str, str]]:
    if processors_packages is None:
        return processors or {}
    elif processors is None:
        raise ValueError("Need to set processors when processors_packages is not None")
    else:
        processors = processors.split(",")
        processors_packages = processors_packages.split(",")
        assert len(processors) == len(processors_packages), "stanza processors and packages doesn't match"
        return dict(zip(processors, processors_packages))


def get_stanza_load_list(lang: str = "en",
                         package: str = "default",
                         processors: Union[str, Dict[str, str]] = {}) -> List[List[str]]:
    stanza_dir = get_stanza_model_dir()
    resources_filepath = os.path.join(stanza_dir, "resources.json")
    with open(resources_filepath) as infile:
        resources = json.load(infile)
    lang, _, package, processors = process_pipeline_parameters(lang, stanza_dir, package, processors)
    stanza_load_list = maintain_processor_list(resources, lang, package, processors)
    return stanza_load_list