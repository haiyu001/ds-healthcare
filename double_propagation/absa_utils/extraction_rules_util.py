from typing import List
from double_propagation.absa.data_types import Relation, RelationTerm
from double_propagation.absa.enumerations import RelCategory, Polarity
import re


def mark_candidate_in_sentence(candidate_text: str, sentence_text: str) -> str:
    sentence_text = re.sub(r"[<>]", " ", sentence_text)
    sentence_text = " ".join(sentence_text.split())
    marked_sentence = re.sub(r"\b{}\b".format(re.escape(candidate_text)),
                             r"<{}>".format(candidate_text.replace("\\", "\\\\")),
                             sentence_text)
    return marked_sentence


def is_valid_relation(relation: Relation) -> bool:
    return any(relation.rel in cat.value for cat in (RelCategory.SUBJ, RelCategory.OBJ, RelCategory.MOD))


def is_equivalent_relation(source_relation: Relation, candidate_relation: Relation) -> bool:
    return (source_relation.rel in RelCategory.MOD.value and candidate_relation.rel in RelCategory.MOD.value) or \
           (source_relation.rel in RelCategory.SUBJ.value and candidate_relation.rel in RelCategory.SUBJ.value) or \
           (source_relation.rel in RelCategory.OBJ.value and candidate_relation.rel in RelCategory.OBJ.value) or \
           (source_relation.rel in RelCategory.SUBJ.value and candidate_relation.rel in RelCategory.OBJ.value) or \
           (source_relation.rel in RelCategory.OBJ.value and candidate_relation.rel in RelCategory.SUBJ.value)


def get_conj_polarity(relation: Relation, relations: List[Relation], polarity: str) -> str:
    gov_id, dep_id = relation.gov.id, relation.dep.id
    start_id, end_id = sorted([gov_id, dep_id])
    for i in range(start_id + 1, end_id):
        if relations[i].dep.text == "but":
            return Polarity.NEG.name if polarity == Polarity.POS.name else Polarity.POS.name
    return polarity


def expand_aspect(aspect_relation: Relation, relations: List[Relation], candidate: str) -> RelationTerm:
    aspect_candidate_term = aspect_relation.gov if candidate == "gov" else aspect_relation.dep
    id = aspect_candidate_term.id
    text = aspect_candidate_term.text
    lemma = aspect_candidate_term.lemma

    for i in range(id - 1, -1, -1):
        if relations[i].rel == "compound" and relations[i].dep.pos != "ADJ":
            text = relations[i].dep.text + " " + text
            lemma = relations[i].dep.lemma + " " + lemma
        else:
            break
    for i in range(id + 1, len(relations), 1):
        if relations[i].rel == "compound" and relations[i].dep.pos != "ADJ":
            text = text + " " + relations[i].dep.text
            lemma = lemma + " " + relations[i].dep.lemma
        else:
            break

    expanded_aspect_term = RelationTerm(text, aspect_candidate_term.pos, lemma, -1)
    return expanded_aspect_term