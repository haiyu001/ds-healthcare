from typing import Optional, Tuple, Dict, List
from annotation.annotation_utils.negation_util import proposition_pseudo, proposition_preceding, proposition_following, \
    proposition_termination, pseudo_clinical, proposition_preceding_clinical, proposition_following_clinical, \
    proposition_termination_clinical, proposition_preceding_clinical_sensitive, proposition_chunk_prefix
from spacy.tokens import Doc, Span
from spacy.matcher import PhraseMatcher
from langcodes import Language


class NegationDetector(object):

    def __init__(self,
                 nlp: Language,
                 neg_termset: str = "en",
                 max_distance: int = 5,
                 entity_types: Optional[str] = None,
                 umls_concept_types: Optional[str] = None,
                 chunk_prefix: Optional[str] = None,
                 pseudo_negations: Optional[str] = None,
                 preceding_negations: Optional[str] = None,
                 following_negations: Optional[str] = None,
                 termination: Optional[str] = None,
                 attrs: Tuple[str] = ("negation",)):
        self.nlp = nlp
        self.neg_termset = neg_termset
        self.max_distance = max_distance
        self.entity_types = entity_types.split(",") if entity_types else None
        self.umls_concept_types = umls_concept_types.split(",") if umls_concept_types else None
        self.chunk_prefix = list(set(proposition_chunk_prefix + chunk_prefix.split(","))) \
            if chunk_prefix else proposition_chunk_prefix
        self.pseudo_negations = pseudo_negations
        self.preceding_negations = preceding_negations
        self.following_negations = following_negations
        self.termination = termination
        self._negation, = attrs
        self.negation_matcher = self.build_negation_matcher()
        Span.set_extension(self._negation, default=False, force=True)

    def __call__(self, doc) -> Doc:
        preceding_negations, following_negations, termination_starts = self.get_negations_and_terminations(doc)
        boundaries = self.get_boundaries(doc, termination_starts)
        for boundary in boundaries:
            preceding_negations_in_boundary = [preceding_negation for preceding_negation in preceding_negations
                                               if boundary[0] <= preceding_negation[0] < boundary[1]]
            following_negations_in_boundary = [following_negation for following_negation in following_negations
                                               if boundary[0] <= following_negation[0] < boundary[1]]
            # set entity negation
            entities_in_boundary = [entity for entity in doc.ents
                                    if entity.start >= boundary[0] and entity.end < boundary[1]]
            if self.entity_types:
                entities_in_boundary = [entity for entity in entities_in_boundary if entity.label_ in self.entity_types]
            self.set_negation_in_boundary(doc, entities_in_boundary, preceding_negations_in_boundary,
                                          following_negations_in_boundary)
            # set concept negation
            umls_concepts_in_boundary = [umls_concept for umls_concept in doc._.umls_concepts
                                         if umls_concept.start >= boundary[0] and umls_concept.end < boundary[1]]
            if self.umls_concept_types:
                umls_concepts_in_boundary = [umls_concept for umls_concept in umls_concepts_in_boundary
                                             if len(set(self.umls_concept_types) &
                                                    {concept["concept_id"] for concept in umls_concept._.concepts}) > 0]
            self.set_negation_in_boundary(doc, umls_concepts_in_boundary, preceding_negations_in_boundary,
                                          following_negations_in_boundary)
        return doc

    def set_negation_in_boundary(self, doc, propositions_in_boundary: List[Span],
                                 preceding_negations_in_boundary: List[Tuple[int, int]],
                                 following_negations_in_boundary: List[Tuple[int, int]]):
        for proposition in propositions_in_boundary:
            if any(0 <= self._get_distance(doc, preceding_negation_end, proposition.start) <= self.max_distance
                   for _, preceding_negation_end in preceding_negations_in_boundary):
                proposition._.set(self._negation, True)
                continue
            if any(0 <= self._get_distance(doc, proposition.end, following_negation_start) <= self.max_distance
                   for following_negation_start, _ in following_negations_in_boundary):
                proposition._.set(self._negation, True)
                continue
            if self.chunk_prefix and any(proposition.text.lower().startswith(c.lower()) for c in self.chunk_prefix):
                proposition._.set(self._negation, True)

    def _get_distance(self, doc, left, right):
        distance = right - left
        if distance > self.max_distance:
            comma_ids = [token.i for token in doc[left: right] if token.text == "," or token.text == "and"]
            distance_ids = sorted(list(set([left, right] + comma_ids)))
            distances = [distance_ids[i + 1] - distance_ids[i] - 1 for i in range(len(distance_ids) - 1)]
            non_zero_distances = [distance for distance in distances if distance > 0]
            if non_zero_distances and all(distance <= 3 for distance in non_zero_distances):
                distance = 0
        return distance

    def _get_neg_terms(self) -> Dict[str, List[str]]:
        if self.neg_termset not in ["en", "en_clinical", "en_clinical_sensitive"]:
            raise ValueError(f"Unsupported neg_temset of {self.neg_termset}")
        neg_terms = {"pseudo_negations": proposition_pseudo,
                     "preceding_negations": proposition_preceding,
                     "following_negations": proposition_following,
                     "termination": proposition_termination, }
        if self.neg_termset.startswith("en_clinical"):
            neg_terms["pseudo_negations"] = pseudo_clinical
            neg_terms["preceding_negations"] = proposition_preceding_clinical
            neg_terms["following_negations"] = proposition_following_clinical
            neg_terms["termination"] = proposition_termination_clinical
        if self.neg_termset == "en_clinical_sensitive":
            neg_terms["preceding_negations"] = proposition_preceding_clinical_sensitive

        if self.pseudo_negations:
            neg_terms["pseudo_negations"] = list(set(neg_terms["pseudo_negations"] + self.pseudo_negations.split(",")))
        if self.preceding_negations:
            neg_terms["preceding_negations"] = \
                list(set(neg_terms["preceding_negations"] + self.preceding_negations.split(",")))
        if self.following_negations:
            neg_terms["following_negations"] = \
                list(set(neg_terms["following_negations"] + self.following_negations.split(",")))
        if self.termination:
            neg_terms["termination"] = list(set(neg_terms["termination"] + self.termination.split(",")))
        return neg_terms

    def build_negation_matcher(self) -> PhraseMatcher:
        neg_terms = self._get_neg_terms()
        base_tokenizer = self.nlp.tokenizer.base_tokenizer
        negation_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        negation_matcher.add("pseudo", list(base_tokenizer.pipe(neg_terms["pseudo_negations"])))
        negation_matcher.add("preceding", list(base_tokenizer.pipe(neg_terms["preceding_negations"])))
        negation_matcher.add("following", list(base_tokenizer.pipe(neg_terms["following_negations"])))
        negation_matcher.add("termination", list(base_tokenizer.pipe(neg_terms["termination"])))
        return negation_matcher

    def get_negations_and_terminations(self, doc: Doc) \
            -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[int]]:
        preceding_negations, following_negations, termination_starts = [], [], []
        matches = self.negation_matcher(doc)
        pseudo = [(start, end) for match_id, start, end in matches if self.nlp.vocab.strings[match_id] == "pseudo"]
        for match_id, start, end in matches:
            if self.nlp.vocab.strings[match_id] == "pseudo" or \
                    any([pseudo_start <= start <= pseudo_end for pseudo_start, pseudo_end in pseudo]):
                continue
            if self.nlp.vocab.strings[match_id] == "preceding":
                preceding_negations.append((start, end))
            elif self.nlp.vocab.strings[match_id] == "following":
                following_negations.append((start, end))
            elif self.nlp.vocab.strings[match_id] == "termination":
                termination_starts.append(start)
        return preceding_negations, following_negations, termination_starts

    def get_boundaries(self, doc: Doc, termination_starts: List[int]) -> List[Tuple[int, int]]:
        sentence_starts = [sent.start for sent in doc.sents]
        starts = sorted(list(set(sentence_starts + termination_starts + [0, len(doc)])))
        boundaries = [(starts[i], starts[i + 1]) for i in range(len(starts) - 1)]
        return boundaries
