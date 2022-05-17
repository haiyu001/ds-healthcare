from typing import Optional, List
from double_propagation.absa.data_types import CandidateTerm, Relation
from double_propagation.absa.enumerations import POS, Polarity, RuleType
from double_propagation.absa_utils.extraction_rules_util import expand_aspect, mark_candidate_in_sentence, \
    is_equivalent_relation, is_valid_relation, get_conj_polarity


def rule_O_O(relation: Relation,
             relations: List[Relation],
             gov_polarity: Optional[str],
             dep_polarity: Optional[str],
             sentence_text: str) -> Optional[CandidateTerm]:
    opinion_source_relation_term, opinion_candidate_relation_term = \
        (relation.gov, relation.dep) if gov_polarity else (relation.dep, relation.gov)
    polarity = gov_polarity or dep_polarity
    if opinion_candidate_relation_term.pos == POS.ADJ.name and relation.rel == "conj":
        return CandidateTerm(opinion_candidate_relation_term.text,
                             opinion_candidate_relation_term.pos,
                             opinion_candidate_relation_term.lemma,
                             get_conj_polarity(relation, relations, polarity),
                             opinion_source_relation_term.text,
                             RuleType.O_O.name,
                             mark_candidate_in_sentence(opinion_candidate_relation_term.text, sentence_text))


def rule_O_X_O(relation: Relation, relations: List[Relation], polarity: str, sentence_text: str) \
        -> Optional[CandidateTerm]:
    for candidate_relation in relations:
        if candidate_relation.gov is relation.gov and relation.rel == candidate_relation.rel and \
                candidate_relation.dep.pos == POS.ADJ.name and candidate_relation is not relation:
            opinion_candidate_relation_term = candidate_relation.dep
            opinion_source_relation_term = relation.dep
            return CandidateTerm(opinion_candidate_relation_term.text,
                                 opinion_candidate_relation_term.pos,
                                 opinion_candidate_relation_term.lemma,
                                 polarity,
                                 opinion_source_relation_term.text,
                                 RuleType.O_X_O.name,
                                 mark_candidate_in_sentence(opinion_candidate_relation_term.text, sentence_text))


def rule_A_O(relation: Relation, sentence_sentiment: str, sentence_text: str) -> Optional[CandidateTerm]:
    opinion_candidate_relation_term = relation.dep
    aspect_source_relation_term = relation.gov
    if is_valid_relation(relation) and opinion_candidate_relation_term.pos == POS.ADJ.name:
        return CandidateTerm(opinion_candidate_relation_term.text,
                             opinion_candidate_relation_term.pos,
                             opinion_candidate_relation_term.lemma,
                             sentence_sentiment,
                             aspect_source_relation_term.text,
                             RuleType.A_O.name,
                             mark_candidate_in_sentence(opinion_candidate_relation_term.text, sentence_text))


def rule_A_X_O(relation: Relation, relations: List[Relation], sentence_sentiment: str, sentence_text: str) \
        -> Optional[CandidateTerm]:
    for candidate_relation in relations:
        if candidate_relation.gov is relation.gov and \
                candidate_relation.dep.pos == POS.ADJ.name and \
                is_valid_relation(relation) and \
                is_valid_relation(candidate_relation) and \
                candidate_relation is not relation:
            opinion_candidate_relation_term = candidate_relation.dep
            aspect_source_relation_term = relation.dep
            return CandidateTerm(opinion_candidate_relation_term.text,
                                 opinion_candidate_relation_term.pos,
                                 opinion_candidate_relation_term.lemma,
                                 sentence_sentiment,
                                 aspect_source_relation_term.text,
                                 RuleType.A_X_O.name,
                                 mark_candidate_in_sentence(opinion_candidate_relation_term.text, sentence_text))


def rule_O_A(relation: Relation, relations: List[Relation], sentence_text: str) -> Optional[CandidateTerm]:
    if relation.gov.pos in ["NN", "NNP"] and is_valid_relation(relation):
        aspect_candidate_relation_term = expand_aspect(relation, relations, "gov")
        opinion_source_relation_term = relation.dep
        return CandidateTerm(aspect_candidate_relation_term.text,
                             aspect_candidate_relation_term.pos,
                             aspect_candidate_relation_term.lemma,
                             Polarity.UNK.name,
                             opinion_source_relation_term.text,
                             RuleType.O_A.name,
                             mark_candidate_in_sentence(aspect_candidate_relation_term.text, sentence_text))


def rule_O_X_A(relation: Relation, relations: List[Relation], sentence_text: str) -> Optional[CandidateTerm]:
    for candidate_relation in relations:
        if candidate_relation.gov is relation.gov and candidate_relation.dep.pos in ["NN", "NNP"] and \
                is_valid_relation(relation) and is_valid_relation(candidate_relation) and \
                candidate_relation is not relation:
            aspect_candidate_relation_term = expand_aspect(candidate_relation, relations, "dep")
            opinion_source_relation_term = relation.dep
            return CandidateTerm(aspect_candidate_relation_term.text,
                                 aspect_candidate_relation_term.pos,
                                 aspect_candidate_relation_term.lemma,
                                 Polarity.UNK.name,
                                 opinion_source_relation_term.text,
                                 RuleType.O_X_A.name,
                                 mark_candidate_in_sentence(aspect_candidate_relation_term.text, sentence_text))


def rule_A_A(relation: Relation, relations: List[Relation], gov_in_aspect_sources: bool, sentence_text: str) \
        -> Optional[CandidateTerm]:
    aspect_candidate_relation_term = relation.dep if gov_in_aspect_sources else relation.gov
    aspect_source_relation_term = relation.gov if gov_in_aspect_sources else relation.dep
    if aspect_candidate_relation_term.pos in ["NN", "NNP"] and relation.rel == "conj":
        candidate = "dep" if gov_in_aspect_sources else "gov"
        aspect_candidate_relation_term = expand_aspect(relation, relations, candidate)
        return CandidateTerm(aspect_candidate_relation_term.text,
                             aspect_candidate_relation_term.pos,
                             aspect_candidate_relation_term.lemma,
                             Polarity.UNK.name,
                             aspect_source_relation_term.text,
                             RuleType.A_A.name,
                             mark_candidate_in_sentence(aspect_candidate_relation_term.text, sentence_text))


def rule_A_X_A(relation: Relation, relations: List[Relation], sentence_text: str) -> Optional[CandidateTerm]:
    for candidate_relation in relations:
        if candidate_relation.gov is relation.gov and candidate_relation.dep.pos in ["NN", "NNP"] and \
                is_equivalent_relation(relation, candidate_relation) and \
                candidate_relation is not relation:
            aspect_candidate_relation_term = expand_aspect(candidate_relation, relations, "dep")
            aspect_source_relation_term = relation.dep
            return CandidateTerm(aspect_candidate_relation_term.text,
                                 aspect_candidate_relation_term.pos,
                                 aspect_candidate_relation_term.lemma,
                                 Polarity.UNK.name,
                                 aspect_source_relation_term.text,
                                 RuleType.A_X_A.name,
                                 mark_candidate_in_sentence(aspect_candidate_relation_term.text, sentence_text))
