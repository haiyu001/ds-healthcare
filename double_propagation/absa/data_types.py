from __future__ import annotations
from typing import NamedTuple, List, Dict, Any, Optional
from pyspark.sql.types import StructField, StringType, StructType, ArrayType


class RelationTerm(NamedTuple):
    text: str
    pos: str
    lemma: str
    id: int


class Relation(NamedTuple):
    gov: RelationTerm
    dep: RelationTerm
    rel: str


class CandidateTerm(NamedTuple):
    text: str
    pos: str
    lemma: str
    polarity: str
    source: str
    rule: str
    sentence: str


class AspectTerm(object):
    def __init__(self, text: str, pos: str):
        self.text = text
        self.pos = pos

    def __eq__(self, other: AspectTerm) -> bool:
        return self.text.lower() == other.text.lower() and self.pos == other.pos

    def __str__(self) -> str:
        return f"{self.text} ({self.pos})"

    @classmethod
    def from_candidate_term(cls, candidate_term: CandidateTerm) -> AspectTerm:
        return AspectTerm(candidate_term.text, candidate_term.pos)


class InferenceAspectTerm(NamedTuple):
    text: str
    start_char: int
    sentiment_score: float
    aspect: str
    hierarchy: str


class InferenceOpinionTerm(NamedTuple):
    text: str
    start_char: int
    sentiment_score: float
    opinion: str
    rule: str
    intensifiers: List[str]
    negations: List[str]


class InferenceTriplet(NamedTuple):
    aspect: InferenceAspectTerm
    opinions: List[InferenceOpinionTerm]


class InferenceDoc(object):
    def __init__(self,
                 doc_text: str,
                 doc_marked_text: str,
                 doc_metadata: Optional[Dict[str, Any]],
                 doc_triplets: List[InferenceTriplet]):
        self.doc_text = doc_text
        self.doc_marked_text = doc_marked_text
        self.doc_metadata = doc_metadata
        self.doc_triplets = doc_triplets

    def to_dict(self):
        doc_triplets_list = []
        for triplet in self.doc_triplets:
            doc_triplets_list.append({"aspect": triplet.aspect._asdict(),
                                      "opinions": [opinion._asdict() for opinion in triplet.opinions]})
        inference_doc_dict = {
            "text": self.doc_text,
            "marked_text": self.doc_marked_text,
            "metadata": self.doc_metadata,
            "triplets": doc_triplets_list
        }
        return inference_doc_dict


candidate_term_schema = StructType([
    StructField("text", StringType()),
    StructField("pos", StringType()),
    StructField("lemma", StringType()),
    StructField("polarity", StringType()),
    StructField("source", StringType()),
    StructField("rule", StringType()),
    StructField("sentence", StringType()),
])

candidates_schema = StructType([
    StructField("opinions", ArrayType(candidate_term_schema)),
    StructField("aspects", ArrayType(candidate_term_schema)),
])
