from __future__ import annotations
from typing import NamedTuple
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
