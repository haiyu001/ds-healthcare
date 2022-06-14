from typing import Dict, Optional, Set, List, Tuple
from pyspark.sql.types import StringType, Row
from annotation.annotation_utils.annotator_util import load_blank_nlp
from annotation.annotation_utils.annotator_spark_util import load_annotation
from double_propagation.absa.data_types import InferenceOpinionTerm, InferenceAspectTerm, InferenceTriplet, InferenceDoc
from double_propagation.absa_utils.triplet_inference_util import load_aspect_hierarchy, get_aspect_matcher, \
    get_mark_sign
from lexicon.negation_lexicon import sentiment_negations, sentiment_negations_social, sentiment_negations_pseudo
from lexicon.canonicalization_lexicon import intensifiers
from pyspark.sql import DataFrame, Column
from spacy import Language
from spacy.tokens import Doc, Span
from spacy.util import filter_spans
from spacy.matcher import Matcher
from itertools import accumulate
import pyspark.sql.functions as F
import statistics
import json


def capital_check(text: str, score: float, cap_scalar: float) -> float:
    if text.isupper():
        if score > 0:
            score += cap_scalar
        else:
            score -= cap_scalar
    return score


def intensifier_check(prev_id: int,
                      opinion_id: int,
                      tokens: List[Row],
                      sentiment_score: float,
                      cap_scalar: float,
                      intensifiers_lexicon: Dict[str, float]) -> Tuple[Optional[str], float]:
    intensifier, intensifier_scalar = None, 0.0
    distance = opinion_id - prev_id
    prev_word = tokens[prev_id].text
    if distance > 1:
        next_word = tokens[prev_id + 1].text
        prev_next_word = prev_word + " " + next_word
        if prev_next_word.lower() in intensifiers_lexicon:
            intensifier, intensifier_scalar = prev_next_word, intensifiers_lexicon[prev_next_word.lower()]
    if intensifier is None and prev_word.lower() in intensifiers_lexicon:
        intensifier, intensifier_scalar = prev_word, intensifiers_lexicon[prev_word.lower()]
    if intensifier:
        if sentiment_score < 0:
            intensifier_scalar *= -1
        intensifier_scalar = capital_check(intensifier, intensifier_scalar, cap_scalar)
        if distance == 2:
            intensifier_scalar *= 0.95
        elif distance == 3:
            intensifier_scalar *= 0.9
        sentiment_score += intensifier_scalar
    return intensifier, sentiment_score


def negation_check(prev_id: int,
                   opinion_id: int,
                   tokens: List[Row],
                   sentiment_score: float,
                   neg_scalar: float,
                   negations_lexicon: Set[str],
                   pseudo_negations_lexicon: Dict[str, float]) -> Tuple[Optional[str], Optional[str], float]:
    negation, intensifier = None, None
    distance = opinion_id - prev_id
    prev_word = tokens[prev_id].text

    if distance == 1:
        if prev_word.lower() in negations_lexicon:
            negation = prev_word
            sentiment_score *= neg_scalar
    elif distance == 2:
        next_word = tokens[prev_id + 1].text
        prev_next_word = prev_word + " " + next_word
        if prev_next_word.lower() in pseudo_negations_lexicon:
            pseudo_negations_scalar = pseudo_negations_lexicon[prev_next_word.lower()]
            sentiment_score *= pseudo_negations_scalar
            if pseudo_negations_scalar > 1.0:
                intensifier = prev_next_word
        elif prev_word.lower() in negations_lexicon:
            negation = prev_word
            sentiment_score *= neg_scalar
    else:
        next_word, last_word = tokens[prev_id + 1].text, tokens[prev_id + 2].text
        prev_next_word, prev_last_word = prev_word + " " + next_word, prev_word + " " + last_word
        if prev_next_word.lower() in pseudo_negations_lexicon:
            pseudo_negations_scalar = pseudo_negations_lexicon[prev_next_word.lower()]
            sentiment_score *= pseudo_negations_scalar
            if pseudo_negations_scalar > 1.0:
                intensifier = prev_next_word
        elif prev_last_word.lower() in pseudo_negations_lexicon:
            pseudo_negations_scalar = pseudo_negations_lexicon[prev_last_word.lower()]
            sentiment_score *= pseudo_negations_scalar
            if pseudo_negations_scalar > 1.0:
                intensifier = prev_last_word
        elif prev_word.lower() in negations_lexicon:
            negation = prev_word
            sentiment_score *= neg_scalar
    return negation, intensifier, sentiment_score


def least_check(opinion_id: int,
                tokens: List[Row],
                sentiment_score: float,
                neg_scalar: float) -> Tuple[Optional[str], float]:
    negation = None
    if opinion_id > 0 and tokens[opinion_id - 1].text.lower() == "least":
        if opinion_id == 1 or tokens[opinion_id - 2].text.lower() not in ["at", "very"]:
            negation = tokens[opinion_id - 1].text.lower()
            sentiment_score *= neg_scalar
    return negation, sentiment_score


def get_sentiment_score(opinion: Row,
                        tokens: List[Row],
                        opinion_to_score: Dict[str, float],
                        negations_lexicon: Set[str],
                        pseudo_negations_lexicon: Dict[str, float],
                        intensifiers_lexicon: Dict[str, float],
                        intensifier_negation_max_distance: int,
                        cap_scalar: float,
                        neg_scalar: float) -> Tuple[float, Optional[List[str]], Optional[List[str]]]:
    opinion_id, opinion_text = opinion.id, opinion.text
    sentiment_score = opinion_to_score[opinion_text.lower()]
    sentiment_score = capital_check(opinion_text, sentiment_score, cap_scalar)
    opinion_intensifiers, opinion_negations = [], []
    for prev_id in range(opinion_id - 1, max(0, opinion_id - intensifier_negation_max_distance - 1), - 1):
        # intensifier check
        intensifier, sentiment_score = intensifier_check(prev_id, opinion_id, tokens, sentiment_score,
                                                         cap_scalar, intensifiers_lexicon)
        if intensifier:
            opinion_intensifiers.append(intensifier)
        # negation check
        negation, intensifier, sentiment_score = negation_check(prev_id, opinion_id, tokens, sentiment_score,
                                                                neg_scalar, negations_lexicon, pseudo_negations_lexicon)
        if negation:
            opinion_negations.append(negation)
        if intensifier:
            opinion_intensifiers.append(intensifier)
        # least check
        negation, sentiment_score = least_check(opinion_id, tokens, sentiment_score, neg_scalar)
        if negation:
            opinion_negations.append(negation)
    opinion_intensifiers = opinion_intensifiers[::-1] if opinion_intensifiers else None
    opinion_negations = opinion_negations[::-1] if opinion_negations else None
    return sentiment_score, opinion_intensifiers, opinion_negations


def get_aspect_opinions(aspect_span: Span,
                        tokens: List[Row],
                        opinion_to_score: Dict[str, float]) -> List[Tuple[Row, str]]:
    aspect_opinions = []
    aspect_span_ids = list(range(aspect_span.end - 1, aspect_span.start - 1, -1))
    for aspect_token_id in aspect_span_ids:
        # try direct dep first
        for token in tokens:
            token_gov = tokens[token.gov]
            if token_gov.id == aspect_token_id and token.text.lower() in opinion_to_score and token.id not in aspect_span_ids:
                aspect_opinions.append((token, "A->O"))
            if token.id == aspect_token_id and token_gov.text.lower() in opinion_to_score and token_gov.id not in aspect_span_ids:
                aspect_opinions.append((token_gov, "O->A"))
        # if direct dep not find any opinion, try indirect dep
        if not aspect_opinions:
            aspect_gov = tokens[tokens[aspect_token_id].gov]
            aspect_gov_gov = tokens[aspect_gov.gov]
            if aspect_gov_gov.text.lower() in opinion_to_score and aspect_gov_gov.id not in aspect_span_ids:
                aspect_opinions.append((aspect_gov_gov, "A<-X<-O"))
            for token in tokens:
                if token.gov == aspect_gov.id and token.id != aspect_token_id and token.text.lower() in opinion_to_score and \
                        token.id not in aspect_span_ids:
                    aspect_opinions.append((token, "A<-X->O"))
        if aspect_opinions:
            break
    return aspect_opinions


def udf_inference(tokens: Column,
                  sentences: Column,
                  metadata: Column,
                  nlp: Language,
                  opinion_to_score: Dict[str, float],
                  aspect_to_hierarchy: Dict[str, str],
                  aspect_matcher: Matcher,
                  negations_lexicon: Set[str],
                  pseudo_negations_lexicon: Dict[str, float],
                  intensifiers_lexicon: Dict[str, float],
                  intensifier_negation_max_distance: int,
                  cap_scalar: float,
                  neg_scalar: float,
                  metadata_fields_to_keep: Optional[str] = None) -> Column:
    def inference(tokens, sentences, metadata):
        doc_org_token_texts_with_ws = [token.org_text_with_ws for token in tokens]
        doc_org_token_start_chars = [0] + list(accumulate([len(i) for i in doc_org_token_texts_with_ws]))[:-1]
        doc_org_text = doc_org_marked_text = "".join(doc_org_token_texts_with_ws)
        sentence_start_ids = {sentence["start_id"] for sentence in sentences}
        doc = Doc(nlp.vocab,
                  words=[token.text for token in tokens],
                  spaces=[token.whitespace for token in tokens],
                  pos=[token.pos for token in tokens],
                  sent_starts=[i in sentence_start_ids for i in range(len(tokens))])
        aspect_spans = filter_spans([doc[start_id: end_id] for _, start_id, end_id in aspect_matcher(doc)])
        aspect_spans = [aspect_span for aspect_span in aspect_spans if all(token.pos_ == "NOUN" or token.pos_ == "PROPN"
                                                                           for token in aspect_span)]
        inference_triplets = []
        mark_offset = 0
        for aspect_span in aspect_spans:
            opinions = get_aspect_opinions(aspect_span, tokens, opinion_to_score)
            inference_opinions = []
            for opinion, opinion_rule in opinions:
                opinion_sentiment_score, opinion_intensifiers, opinion_negations = \
                    get_sentiment_score(opinion, tokens, opinion_to_score, negations_lexicon, pseudo_negations_lexicon,
                                        intensifiers_lexicon, intensifier_negation_max_distance, cap_scalar, neg_scalar)
                opinion_text = opinion.text.lower()
                opinion_org_text = opinion.org_text_with_ws.strip()
                opinion_org_start_char = doc_org_token_start_chars[opinion.id]
                inference_opinions.append(InferenceOpinionTerm(opinion_org_text,
                                                               opinion_org_start_char,
                                                               opinion_sentiment_score,
                                                               opinion_text,
                                                               opinion_rule,
                                                               opinion_intensifiers,
                                                               opinion_negations))

            aspect_org_start_char = doc_org_token_start_chars[aspect_span.start]
            aspect_org_end_char = doc_org_token_start_chars[aspect_span.end]
            if doc_org_token_texts_with_ws[aspect_span.end - 1].endswith(" "):
                aspect_org_end_char -= 1
            aspect_org_text = doc_org_text[aspect_org_start_char: aspect_org_end_char]
            aspect_sentiment_score = statistics.mean([i.sentiment_score for i in inference_opinions]) \
                if inference_opinions else 0.0
            aspect_text = " ".join([token.text.lower() for token in aspect_span])
            aspect_hierarchy = aspect_to_hierarchy[aspect_text]

            left_mark, right_mark = get_mark_sign(aspect_sentiment_score)
            doc_org_marked_text = doc_org_marked_text[:aspect_org_start_char + mark_offset] + \
                                  left_mark + aspect_text + right_mark + \
                                  doc_org_marked_text[aspect_org_end_char + mark_offset:]
            mark_offset += 4

            inference_aspect = InferenceAspectTerm(aspect_org_text,
                                                   aspect_org_start_char,
                                                   aspect_sentiment_score,
                                                   aspect_text,
                                                   aspect_hierarchy)
            if inference_opinions:
                inference_triplets.append(InferenceTriplet(inference_aspect, inference_opinions))

        if inference_triplets:
            doc_metadata = {field: metadata[field] for field in metadata_fields_to_keep.split(",")} \
                if metadata_fields_to_keep else None
            inference_doc = InferenceDoc(doc_org_text, doc_org_marked_text, doc_metadata, inference_triplets)
            inference_doc_json = json.dumps(inference_doc.to_dict(), ensure_ascii=False)
            return inference_doc_json
        else:
            return None

    return F.udf(inference, StringType())(tokens, sentences, metadata)


def extract_triplet(annotation_sdf: DataFrame,
                    aspect_filepath: str,
                    opinion_filepath: str,
                    save_folder_dir: str,
                    save_folder_name: str,
                    lang: str,
                    spacy_package: str,
                    social: bool,
                    intensifier_negation_max_distance: int,
                    cap_scalar: float,
                    neg_scalar: float,
                    metadata_fields_to_keep: Optional[str] = None):
    nlp = load_blank_nlp(lang, spacy_package)
    opinion_to_score = load_json_file(opinion_filepath)
    aspect_to_hierarchy = load_aspect_hierarchy(aspect_filepath)
    aspect_matcher = get_aspect_matcher(nlp, list(aspect_to_hierarchy.keys()))
    negations_lexicon = set(sentiment_negations) if not social else set(sentiment_negations_social)
    pseudo_negations_lexicon = sentiment_negations_pseudo
    intensifiers_lexicon = intensifiers
    inference_sdf = annotation_sdf.select(udf_inference(F.col("tokens"),
                                                        F.col("sentences"),
                                                        F.col("_").metadata,
                                                        nlp,
                                                        opinion_to_score,
                                                        aspect_to_hierarchy,
                                                        aspect_matcher,
                                                        negations_lexicon,
                                                        pseudo_negations_lexicon,
                                                        intensifiers_lexicon,
                                                        intensifier_negation_max_distance,
                                                        cap_scalar,
                                                        neg_scalar,
                                                        metadata_fields_to_keep).alias("inference"))
    write_sdf_to_dir(inference_sdf, save_folder_dir, save_folder_name, file_format="txt")


if __name__ == "__main__":
    from utils.general_util import setup_logger, load_json_file
    from utils.config_util import read_config_to_dict
    from utils.resource_util import get_repo_dir, get_data_filepath
    from utils.spark_util import get_spark_session, write_sdf_to_dir
    import os

    setup_logger()

    absa_config_filepath = os.path.join(get_repo_dir(), "double_propagation", "pipelines", "conf/absa_template.cfg")
    absa_config = read_config_to_dict(absa_config_filepath)

    domain_dir = get_data_filepath(absa_config["domain"])
    absa_dir = os.path.join(domain_dir, absa_config["absa_folder"])
    annotation_dir = os.path.join(domain_dir, absa_config["annotation_folder"])
    inference_dir = os.path.join(domain_dir, absa_config["inference_folder"])
    aspect_filepath = os.path.join(absa_dir, absa_config["aspect_filename"])
    opinion_filepath = os.path.join(absa_dir, absa_config["opinion_filename"])

    spark_cores = 4
    spark = get_spark_session("test", master_config=f"local[{spark_cores}]", log_level="Warn")

    annotation_sdf = load_annotation(spark, annotation_dir, absa_config["drop_non_english"])

    extract_triplet(annotation_sdf,
                    aspect_filepath,
                    opinion_filepath,
                    inference_dir,
                    absa_config["absa_inference_folder"],
                    absa_config["lang"],
                    absa_config["spacy_package"],
                    absa_config["social"],
                    absa_config["intensifier_negation_max_distance"],
                    absa_config["cap_scalar"],
                    absa_config["neg_scalar"],
                    absa_config["metadata_fields_to_keep"])
