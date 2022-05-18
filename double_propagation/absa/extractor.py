from typing import Optional, List, Dict, Iterator, Tuple, Set
from pyspark.sql import functions as F, Column
from pyspark.sql.types import Row, BooleanType, StringType, MapType, IntegerType
from annotation.tokenization.preprocessor import REPLACE_EMAIL, REPLACE_URL, REPLACE_HASHTAG, REPLACE_HANDLE
from double_propagation.absa.enumerations import Polarity, RuleType
from double_propagation.absa.extraction_rules import rule_O_O, rule_O_X_O, rule_O_A, rule_O_X_A, rule_A_A, rule_A_O, \
    rule_A_X_O, rule_A_X_A
from double_propagation.absa.data_types import RelationTerm, Relation, AspectTerm, candidates_schema
from double_propagation.absa_utils.extractor_util import load_absa_stop_words, load_absa_seed_opinions, norm_pos, \
    VALID_OPINION_REX, VALID_ASPECT_REX, get_sentence_sentiment, load_word_to_lemma
import pandas as pd
from pyspark.sql import DataFrame
from scipy.stats import hmean
from collections import Counter
import random
import string
import logging
import json
import re

from double_propagation.sentiment_subjectivity.model_building import get_sentiment_features_pdf, get_model_prediction_pdf
from word_vector.wv_space import ConceptNetWordVec, load_txt_vecs_to_pdf


def udf_extract_opinions_and_aspects(doc_text: Column,
                                     doc_sentences: Column,
                                     doc_tokens: Column,
                                     doc_sentiments: Column,
                                     aspects: List[AspectTerm],
                                     opinions: Dict[str, str]) -> Column:
    def extract_opinoins_and_aspects(doc_text, doc_sentences, doc_tokens, doc_sentiments):
        opinion_candidates = []
        aspect_candidates = []
        for sentence_text, sentence_sentiment, relations in \
                iter_sentences(doc_text, doc_sentences, doc_tokens, doc_sentiments):
            for relation in relations:
                if relation.rel == "ROOT":
                    continue
                gov_polarity = opinions.get(relation.gov.text.lower())
                dep_polarity = opinions.get(relation.dep.text.lower())
                if bool(gov_polarity) ^ bool(dep_polarity):
                    opinion_candidates.append(rule_O_O(relation, relations, gov_polarity, dep_polarity, sentence_text))
                if not gov_polarity and dep_polarity:
                    opinion_candidates.append(rule_O_X_O(relation, relations, dep_polarity, sentence_text))
                    aspect_candidates.append(rule_O_A(relation, relations, sentence_text))
                    aspect_candidates.append(rule_O_X_A(relation, relations, sentence_text))
                if not gov_polarity and not dep_polarity:
                    gov_in_aspect = AspectTerm(relation.gov.text, relation.gov.pos) in aspects
                    dep_in_aspect = AspectTerm(relation.dep.text, relation.dep.pos) in aspects
                    if gov_in_aspect ^ dep_in_aspect:
                        aspect_candidates.append(rule_A_A(relation, relations, gov_in_aspect, sentence_text))
                    if gov_in_aspect and not dep_in_aspect:
                        opinion_candidates.append(rule_A_O(relation, sentence_sentiment, sentence_text))
                    if not gov_in_aspect and dep_in_aspect:
                        opinion_candidates.append(rule_A_X_O(relation, relations, sentence_sentiment, sentence_text))
                        aspect_candidates.append(rule_A_X_A(relation, relations, sentence_text))
        opinion_candidates = [opinion_candidate for opinion_candidate in opinion_candidates if opinion_candidate]
        aspect_candidates = [aspect_candidate for aspect_candidate in aspect_candidates if aspect_candidate]
        return opinion_candidates, aspect_candidates

    return F.udf(extract_opinoins_and_aspects, candidates_schema)(doc_text, doc_sentences, doc_tokens, doc_sentiments)


def udf_filter_opinion_candidates(opinion_candidate_term: Column,
                                  opinion_stop_words: Set[str],
                                  accumulated_opinions: Dict[str, str]) -> Column:
    def filter_opinion_candidates(opinion_candidate_term):
        opinion_candidate_text = opinion_candidate_term.text
        return re.match(VALID_OPINION_REX, opinion_candidate_text.lower()) and \
               opinion_candidate_text.lower() not in opinion_stop_words and \
               opinion_candidate_text.lower() not in accumulated_opinions and \
               (opinion_candidate_text.islower() or opinion_candidate_text.isupper())

    return F.udf(filter_opinion_candidates, BooleanType())(opinion_candidate_term)


def udf_filter_aspect_candidates(aspect_candidate_term: Column,
                                 aspect_stop_words: Set[str],
                                 accumulated_aspects: List[AspectTerm]) -> Column:
    def filter_aspect_candidates(aspect_candidate_term):
        aspect_candidate_text = aspect_candidate_term.text
        return re.match(VALID_ASPECT_REX, aspect_candidate_text.lower()) and \
               aspect_candidate_text.lower() not in aspect_stop_words and \
               AspectTerm.from_candidate_term(aspect_candidate_term) not in accumulated_aspects

    return F.udf(filter_aspect_candidates, BooleanType())(aspect_candidate_term)


def udf_get_opinion_polarity(polarities: Column, polarity_filter_min_ratio: float) -> Column:
    def get_opinoin_polarity(polarities):
        pos_count, neg_count, unk_count = 0, 0, 0
        for polarity in polarities:
            if polarity == Polarity.POS.name:
                pos_count += 1
            elif polarity == Polarity.NEG.name:
                neg_count += 1
            else:
                unk_count += 1
        if pos_count > 0 and (neg_count == 0 or (pos_count / neg_count) >= polarity_filter_min_ratio):
            return Polarity.POS.name
        elif neg_count > 0 and (pos_count == 0 or (neg_count / pos_count) >= polarity_filter_min_ratio):
            return Polarity.NEG.name
        else:
            return Polarity.UNK.name

    return F.udf(get_opinoin_polarity, StringType())(polarities)


def udf_get_canonical_text(sentences: Column, text: Column) -> Column:
    def get_canonical_text(sentences, text):
        texts = []
        for sentence in sentences:
            sentence = sentence.strip()
            groups = re.finditer("<(.+?)>", sentence)
            for group in groups:
                start_id, text_candidate = group.start(), group.group()[1:-1]
                if start_id - 2 >= 0 and sentence[start_id - 1] == " " \
                        and sentence[start_id - 2] not in string.punctuation:
                    texts.append(text_candidate)
        canonical_text = Counter(texts).most_common(1)[0][0] if texts else text
        return canonical_text

    return F.udf(get_canonical_text, StringType())(sentences, text)


def udf_get_aspect_opinions(rules: Column, sources: Column, threshold: int) -> Column:
    def get_aspect_opinions(rules, sources):
        opinions = [source.lower() for rule, source in zip(rules, sources) if rule in
                    [RuleType.O_A.name, RuleType.O_X_A.name]]
        opinion_to_count = {k: v for k, v in Counter(opinions).most_common() if v >= threshold}
        return opinion_to_count

    return F.udf(get_aspect_opinions, MapType(StringType(), IntegerType(), False))(rules, sources)


def udf_get_opinion_aspects(rules: Column, sources: Column, threshold: int) -> Column:
    def get_opinion_aspects(rules, sources):
        aspects = [source.lower() for rule, source in zip(rules, sources) if
                   rule in [RuleType.A_O.name, RuleType.A_X_O.name]]
        aspect_to_count = {k: v for k, v in Counter(aspects).most_common() if v >= threshold}
        return aspect_to_count

    return F.udf(get_opinion_aspects, MapType(StringType(), IntegerType(), False))(rules, sources)


def _normalize_underscore_ampersand_lower_lemma(lower_lemma: str, word_to_lemma: Dict[str, str]) -> str:
    if "_" in lower_lemma or "&" in lower_lemma:
        words = re.split(r"_|&", lower_lemma)
        separator = "_" if "_" in lower_lemma else "&"
        lower_lemma = separator.join([word_to_lemma.get(word, word) for word in words])
    return lower_lemma


def load_absa_sdf(annotation_sdf: DataFrame) -> DataFrame:
    absa_columns = [F.col("text").alias("doc_text"),
                    F.col("tokens").alias("doc_tokens"),
                    F.col("sentences").alias("doc_sentences")]
    custom_data = annotation_sdf.select(F.col("_")).schema.jsonValue()
    custom_data_fields = [field["name"] for field in custom_data["fields"][0]["type"]["fields"]]
    if "sentence_sentiments" in custom_data_fields:
        absa_columns.append(F.col("_").sentence_sentiments.alias("doc_sentiments"))
    absa_sdf = annotation_sdf.select(*absa_columns).drop_duplicates(subset=["doc_text"])
    logging.info(f"\n{'=' * 100}\nnum records: {absa_sdf.count()}\n{'=' * 100}\n")
    return absa_sdf


def get_aspect_opinion_thresholds(absa_sdf: DataFrame,
                                  aspect_threshold: int,
                                  opinion_threshold: int,
                                  sentence_filter_min_count: Optional[int]) -> Tuple[int, int]:
    if sentence_filter_min_count:
        domain_sentences_count = absa_sdf.select((F.sum(F.size(F.col("doc_sentences"))))).collect()[0][0]
        sentence_threshold = int(domain_sentences_count / sentence_filter_min_count)
        aspect_threshold = opinion_threshold = sentence_threshold
        logging.info(
            f"\n{'=' * 100}\nnum sentences: {domain_sentences_count}\n{'=' * 100}\n")
    logging.info(
        f"\n{'=' * 100}\naspect_threshold: {aspect_threshold}\topinion_threshold: {opinion_threshold}\n{'=' * 100}\n")
    return aspect_threshold, opinion_threshold


def save_candidates_pdf(candidates_pdf: pd.DataFrame, save_filepath: str):
    preprocessor_replace_list = "|".join([REPLACE_EMAIL, REPLACE_URL, REPLACE_HASHTAG, REPLACE_HANDLE])
    candidates_pdf = candidates_pdf[~(candidates_pdf["text"].str.contains(preprocessor_replace_list, regex=True))]
    candidates_pdf = candidates_pdf.sort_values(by='count', ascending=False)
    save_pdf(candidates_pdf, save_filepath)


def filter_aspect_candidates(aspect_candidates_sdf: DataFrame,
                             aspect_stop_words: Set[str],
                             accumulated_aspects: List[AspectTerm]) -> DataFrame:
    aspect_candidates_sdf = aspect_candidates_sdf.filter(
        udf_filter_aspect_candidates(aspect_candidates_sdf.candidate, aspect_stop_words, accumulated_aspects))
    aspect_candidates_sdf = aspect_candidates_sdf.select(F.lower(F.col("candidate.text")).alias("text"),
                                                         F.col("candidate.pos").alias("pos"),
                                                         F.col("candidate.lemma").alias("lemma"),
                                                         F.col("candidate.source").alias("source"),
                                                         F.col("candidate.rule").alias("rule"),
                                                         F.col("candidate.sentence").alias("sentence"))
    aspect_candidates_sdf = aspect_candidates_sdf.drop_duplicates(["text", "source", "sentence"])
    return aspect_candidates_sdf


def filter_opinion_candidates(opinion_candidates_sdf: DataFrame,
                              opinion_stop_words: Set[str],
                              accumulated_opinions: Dict[str, str]) -> DataFrame:
    opinion_candidates_sdf = opinion_candidates_sdf.filter(
        udf_filter_opinion_candidates(opinion_candidates_sdf.candidate, opinion_stop_words, accumulated_opinions))
    opinion_candidates_sdf = opinion_candidates_sdf.select(F.lower(F.col("candidate.text")).alias("text"),
                                                           F.col("candidate.polarity").alias("polarity"),
                                                           F.col("candidate.source").alias("source"),
                                                           F.col("candidate.rule").alias("rule"),
                                                           F.col("candidate.sentence").alias("sentence"))
    opinion_candidates_sdf = opinion_candidates_sdf.drop_duplicates(["text", "source", "sentence"])
    return opinion_candidates_sdf


def get_next_iter_aspects(aspect_candidates_sdf: DataFrame, aspect_threshold: int) -> List[AspectTerm]:
    aspect_candidates_sdf = aspect_candidates_sdf.groupby("text", "pos").count()
    aspect_candidates_sdf = aspect_candidates_sdf.filter(F.col("count") >= aspect_threshold)
    aspect_candidates_sdf = aspect_candidates_sdf.select("text", "pos")
    aspect_candidates_sdf.cache()
    aspect_candidates_pdf = aspect_candidates_sdf.toPandas()
    next_iter_aspects = [AspectTerm(text, pos) for text, pos in
                         zip(aspect_candidates_pdf["text"].tolist(), aspect_candidates_pdf["pos"].tolist())]
    return next_iter_aspects


def get_next_iter_opinions(opinion_candidates_sdf: DataFrame,
                           opinion_threshold: int,
                           polarity_filter_min_ratio: float) -> Dict[str, str]:
    opinion_candidates_sdf = opinion_candidates_sdf.groupby(["text"]) \
        .agg(F.collect_list("polarity").alias("polarities"), F.count(F.col("polarity")).alias("count"))
    opinion_candidates_sdf = opinion_candidates_sdf.filter(F.col("count") >= opinion_threshold)
    opinion_candidates_sdf = opinion_candidates_sdf.select(
        "text", udf_get_opinion_polarity(F.col("polarities"), polarity_filter_min_ratio).alias("polarity"))
    opinion_candidates_sdf.cache()
    opinion_candidates_pdf = opinion_candidates_sdf.toPandas()
    next_iter_opinoins = dict(zip(opinion_candidates_pdf["text"].tolist(), opinion_candidates_pdf["polarity"].tolist()))
    return next_iter_opinoins


def get_aspect_candidates_pdf(aspect_candidates_sdfs: List[DataFrame],
                              aspect_stop_words: Set[str],
                              aspect_threshold: int,
                              aspect_opinions_filter_min_count: int = 3,
                              aspect_opinions_num_samples: int = 10) -> pd.DataFrame:
    aspect_candidates_sdf = union_sdfs(*aspect_candidates_sdfs)
    aspect_candidates_sdf = filter_aspect_candidates(aspect_candidates_sdf, aspect_stop_words, [])
    aspect_candidates_sdf = aspect_candidates_sdf.groupby(["text"]).agg(F.collect_list("lemma").alias("lemmas"),
                                                                        F.collect_list("rule").alias("rules"),
                                                                        F.collect_list("source").alias("sources"),
                                                                        F.collect_set("sentence").alias("sentences"),
                                                                        F.count("*").alias("count"))
    aspect_candidates_sdf = aspect_candidates_sdf.filter(F.col("count") >= aspect_threshold)
    aspect_candidates_sdf = aspect_candidates_sdf.select(
        udf_get_canonical_text(F.col("sentences"), F.col("text")).alias("text"),
        "count",
        pudf_get_most_common_text(F.col("lemmas")).alias("lemma"),
        udf_get_aspect_opinions(F.col("rules"), F.col("sources"), aspect_opinions_filter_min_count).alias("opinions"),
        F.slice(F.shuffle(F.col("sentences")), 1, aspect_opinions_num_samples).alias("samples"))
    aspect_candidates_pdf = aspect_candidates_sdf.toPandas()
    aspect_candidates_pdf["opinions"] = aspect_candidates_pdf["opinions"].apply(json.dumps, ensure_ascii=False)
    aspect_candidates_pdf["samples"] = aspect_candidates_pdf["samples"].apply(list) \
        .apply(json.dumps, ensure_ascii=False)
    return aspect_candidates_pdf


def get_opinion_candidates_pdf(opinion_candidates_sdfs: List[DataFrame],
                               stop_words: Set[str],
                               opinion_threshold: int,
                               polarity_filter_min_ratio: float,
                               opinion_aspects_filter_min_count: int = 3,
                               opinion_aspects_num_samples: int = 10) -> pd.DataFrame:
    opinion_raw_df = union_sdfs(*opinion_candidates_sdfs)
    opinion_raw_df = filter_opinion_candidates(opinion_raw_df, stop_words, [])
    opinion_raw_df = opinion_raw_df.groupby(["text"]).agg(F.collect_list("polarity").alias("polarities"),
                                                          F.collect_list("rule").alias("rules"),
                                                          F.collect_list("source").alias("sources"),
                                                          F.collect_set("sentence").alias("sentences"),
                                                          F.count("*").alias("count"))
    opinion_raw_df = opinion_raw_df.filter(F.col("count") >= opinion_threshold)
    opinion_raw_df = opinion_raw_df.select(
        "text",
        "count",
        udf_get_opinion_polarity(F.col("polarities"), polarity_filter_min_ratio).alias("polarity"),
        udf_get_opinion_aspects(F.col("rules"), F.col("sources"), opinion_aspects_filter_min_count).alias("aspects"),
        F.slice(F.shuffle(F.col("sentences")), 1, opinion_aspects_num_samples).alias("samples"))
    opinion_candidates_pdf = opinion_raw_df.toPandas()
    opinion_candidates_pdf["aspects"] = opinion_candidates_pdf["aspects"].apply(json.dumps, ensure_ascii=False)
    opinion_candidates_pdf["samples"] = opinion_candidates_pdf["samples"].apply(list) \
        .apply(json.dumps, ensure_ascii=False)
    return opinion_candidates_pdf


def save_aspect_merge_pdf(aspect_candidates_filepath: str,
                          aspect_merge_filepath: str,
                          unigram_filepath: str,
                          aspect_opinion_num_samples: int):
    aspect_candidates_pdf = pd.read_csv(
        aspect_candidates_filepath, encoding="utf-8", na_values="", keep_default_na=False)
    aspect_candidates_pdf["samples"] = aspect_candidates_pdf["samples"].apply(json.loads)
    aspect_candidates_pdf["lower_lemma"] = aspect_candidates_pdf["lemma"].str.lower()
    aspect_candidates_pdf["lower_lemma"] = aspect_candidates_pdf["lower_lemma"].apply(
        _normalize_underscore_ampersand_lower_lemma, word_to_lemma=load_word_to_lemma(unigram_filepath))
    aspect_lemma_merge_list = []
    for lower_lemma, aspect_group_pdf in aspect_candidates_pdf.groupby("lower_lemma"):
        members = aspect_group_pdf["text"].tolist()
        text = sorted(members, key=len)[0]
        count = sum(aspect_group_pdf["count"])
        samples = sum(aspect_group_pdf["samples"].tolist(), [])
        samples = random.sample(samples, min(aspect_opinion_num_samples, len(samples)))
        non_lower_lemmas = [lemma for lemma in aspect_group_pdf["lemma"] if not lemma.islower()]
        lemma = non_lower_lemmas[0] if non_lower_lemmas else lower_lemma
        aspect_lemma_merge_list.append({"text": text,
                                        "members": json.dumps(members, ensure_ascii=False),
                                        "count": count,
                                        "lemma": lemma,
                                        "samples": json.dumps(samples, ensure_ascii=False)})
    aspect_lemma_merge_pdf = pd.DataFrame(aspect_lemma_merge_list).sort_values(by="count", ascending=False)
    save_pdf(aspect_lemma_merge_pdf, aspect_merge_filepath)


def save_opinion_rank_pdf(opinion_candidates_filepath: str,
                          opinion_vecs_filepath: str,
                          opinion_rank_filepath: str):
    opinion_candidates_pdf = pd.read_csv(opinion_candidates_filepath, index_col="text", encoding="utf-8",
                                         keep_default_na=False, na_values="")
    opinions = opinion_candidates_pdf.index.tolist()
    # extract opinion vecs
    conceptnet_vecs_filepath = get_model_filepath("model", "conceptnet", "numberbatch-en-19.08.txt")
    conceptnet_wordvec = ConceptNetWordVec(conceptnet_vecs_filepath, use_oov_strategy=True)
    conceptnet_wordvec.extract_txt_vecs(opinions, opinion_vecs_filepath)
    # run sentiment model prediction
    sentiment_features_pdf = get_sentiment_features_pdf(opinion_vecs_filepath)
    sentiment_model_filepath = get_model_filepath("model", "sentiment", "sentiment.hdf5")
    opinion_sentiment_scores_pdf = get_model_prediction_pdf(sentiment_features_pdf, sentiment_model_filepath,
                                                            predicted_score_col="sentiment_score")
    # run subjectivity model prediction
    subjectivity_features_pdf = load_txt_vecs_to_pdf(opinion_vecs_filepath)
    subjectivity_model_filepath = get_model_filepath("model", "subjectivity", "subjectivity.hdf5")
    opinion_subjectivity_scores_pdf = get_model_prediction_pdf(subjectivity_features_pdf, subjectivity_model_filepath,
                                                               predicted_score_col="subjectivity_score")
    # get opinion sentiment subjectivity scores
    opinion_sentiment_subjectivity_scores_pdf = \
        opinion_sentiment_scores_pdf.merge(opinion_subjectivity_scores_pdf, on="word").rename(columns={"word": "text"})
    opinion_sentiment_subjectivity_scores_pdf["hmean_score"] = hmean(opinion_sentiment_subjectivity_scores_pdf, axis=1)

    opinion_rank_pdf = pd.concat([opinion_sentiment_subjectivity_scores_pdf,
                                  sentiment_features_pdf[["neg_avg", "pos_avg"]],
                                  opinion_candidates_pdf], axis=1)
    opinion_rank_pdf = opinion_rank_pdf[
        ["count", "polarity", "neg_avg", "pos_avg", "sentiment_score", "subjectivity_score", "hmean_score", "samples"]]
    opinion_rank_pdf = opinion_rank_pdf.sort_values(by="hmean_score", ascending=False)
    save_pdf(opinion_rank_pdf, opinion_rank_filepath, csv_index_label="text", csv_index=True)


def iter_sentences(doc_text: str,
                   doc_sentences: List[Row],
                   doc_tokens: List[Row],
                   doc_sentiments: List[int]) -> Iterator[Tuple[str, str, List[Relation]]]:
    relation_terms = dict()
    if not doc_sentiments:
        doc_sentiments = [Polarity.UNK.name] * len(doc_sentences)
    for sentence, sentence_sentiment in zip(doc_sentences, doc_sentiments):
        sentence_start_id, sentence_end_id = sentence["start_id"], sentence["end_id"]
        sentence_text = doc_text[doc_tokens[sentence_start_id].start_char: doc_tokens[sentence_end_id - 1].end_char]
        sentence_relations = []
        for i in range(sentence_start_id, sentence_end_id):
            sent_token = doc_tokens[i]
            sent_token_gov_id = sent_token.gov
            if sent_token_gov_id not in relation_terms:
                sent_token_gov = doc_tokens[sent_token_gov_id]
                relation_terms[sent_token_gov_id] = RelationTerm(sent_token_gov.text,
                                                                 norm_pos(sent_token_gov.tag, sent_token_gov.pos),
                                                                 sent_token_gov.lemma,
                                                                 sent_token_gov.id - sentence_start_id)
            sent_token_id = sent_token.id
            if sent_token_id not in relation_terms:
                relation_terms[sent_token_id] = RelationTerm(sent_token.text,
                                                             norm_pos(sent_token.tag, sent_token.pos),
                                                             sent_token.lemma,
                                                             sent_token.id - sentence_start_id)
            sentence_relations.append(Relation(relation_terms[sent_token_gov_id],
                                               relation_terms[sent_token_id],
                                               sent_token.rel))
        yield sentence_text, get_sentence_sentiment(sentence_sentiment), sentence_relations


def extract_candidates(annotation_sdf: DataFrame,
                       aspect_candidates_filepath: str,
                       opinion_candidates_filepath: str,
                       aspect_threshold: int,
                       opinion_threshold: int,
                       sentence_filter_min_count: Optional[int] = None,
                       polarity_filter_min_ratio: float = 2.0,
                       aspect_opinion_filter_min_count: int = 3,
                       aspect_opinion_num_samples: int = 10,
                       max_iterations: int = 3):
    absa_sdf = load_absa_sdf(annotation_sdf)
    aspect_threshold, opinion_threshold = \
        get_aspect_opinion_thresholds(absa_sdf, aspect_threshold, opinion_threshold, sentence_filter_min_count)

    seed_opinions = load_absa_seed_opinions()
    aspect_stop_words = load_absa_stop_words()
    opinion_stop_words = aspect_stop_words | seed_opinions.keys()
    iter_aspects, accumulated_aspects = [], []
    iter_opinions, accumulated_opinions = seed_opinions, {}
    aspect_candidates_sdfs, opinion_candidates_sdfs = [], []

    for i in range(max_iterations):
        logging.info(f"\n{'=' * 100}\niteration {i + 1}\n{'=' * 100}\n")
        candidates_sdf = absa_sdf.select(udf_extract_opinions_and_aspects(
            F.col("doc_text"),
            F.col("doc_sentences"),
            F.col("doc_tokens"),
            F.col("doc_sentiments") if "doc_sentiments" in absa_sdf.columns else F.array(),
            iter_aspects,
            iter_opinions).alias("candidates"))
        candidates_sdf.cache()

        aspect_candidates_sdf = candidates_sdf.select(F.explode(F.col("candidates.aspects")).alias("candidate"))
        opinion_candidates_sdf = candidates_sdf.select(F.explode(F.col("candidates.opinions")).alias("candidate"))
        aspect_candidates_sdfs.append(aspect_candidates_sdf)
        opinion_candidates_sdfs.append(opinion_candidates_sdf)

        filtered_aspect_candidates_sdf = \
            filter_aspect_candidates(aspect_candidates_sdf, aspect_stop_words, accumulated_aspects)
        filtered_opinion_candidates_sdf = \
            filter_opinion_candidates(opinion_candidates_sdf, opinion_stop_words, accumulated_opinions)
        iter_aspects = get_next_iter_aspects(filtered_aspect_candidates_sdf, aspect_threshold)
        iter_opinions = get_next_iter_opinions(filtered_opinion_candidates_sdf, opinion_threshold,
                                               polarity_filter_min_ratio)
        accumulated_aspects.extend(iter_aspects)
        accumulated_opinions.update(iter_opinions)
        logging.info(f"\n{'=' * 100}\nextracted aspects: {len(iter_aspects)}\t"
                     f"extracted opinions: {len(iter_opinions)}\n{'=' * 100}\n")

    aspect_candidates_pdf = get_aspect_candidates_pdf(aspect_candidates_sdfs, aspect_stop_words, aspect_threshold,
                                                      aspect_opinion_filter_min_count, aspect_opinion_num_samples)
    opinion_candidates_pdf = get_opinion_candidates_pdf(opinion_candidates_sdfs, opinion_stop_words,
                                                        opinion_threshold, polarity_filter_min_ratio,
                                                        aspect_opinion_filter_min_count, aspect_opinion_num_samples)
    save_candidates_pdf(aspect_candidates_pdf, aspect_candidates_filepath)
    save_candidates_pdf(opinion_candidates_pdf, opinion_candidates_filepath)


if __name__ == "__main__":
    from utils.general_util import setup_logger, save_pdf, make_dir
    from annotation.components.annotator import load_annotation
    from utils.config_util import read_config_to_dict
    from utils.resource_util import get_repo_dir, get_data_filepath, get_model_filepath
    from utils.spark_util import get_spark_session, union_sdfs, pudf_get_most_common_text
    import os

    setup_logger()

    absa_config_filepath = os.path.join(get_repo_dir(), "double_propagation", "pipelines", "conf/absa_template.cfg")
    absa_config = read_config_to_dict(absa_config_filepath)

    domain_dir = get_data_filepath(absa_config["domain"])
    absa_dir = os.path.join(domain_dir, absa_config["absa_folder"])
    annotation_dir = os.path.join(domain_dir, absa_config["annotation_folder"])
    extraction_dir = os.path.join(domain_dir, absa_config["extraction_folder"])
    absa_aspect_dir = make_dir(os.path.join(absa_dir, "aspect"))
    absa_opinion_dir = make_dir(os.path.join(absa_dir, "opinion"))
    aspect_candidates_filepath = os.path.join(absa_aspect_dir, absa_config["aspect_candidates_filename"])
    opinion_candidates_filepath = os.path.join(absa_opinion_dir, absa_config["opinion_candidates_filename"])
    aspect_merge_filepath = os.path.join(absa_aspect_dir, absa_config["aspect_merge_filename"])
    opinion_vecs_filepath = os.path.join(absa_opinion_dir, absa_config["opinion_vecs_filename"])
    opinion_rank_filepath = os.path.join(absa_opinion_dir, absa_config["opinion_rank_filename"])
    unigram_filepath = os.path.join(extraction_dir, absa_config["unigram_filename"])

    spark_cores = 4
    spark = get_spark_session("test", master_config=f"local[{spark_cores}]", log_level="Warn")

    annotation_sdf = load_annotation(spark, annotation_dir, absa_config["drop_non_english"])

    extract_candidates(annotation_sdf,
                       aspect_candidates_filepath,
                       opinion_candidates_filepath,
                       aspect_threshold=absa_config["aspect_threshold"],
                       opinion_threshold=absa_config["aspect_threshold"],
                       sentence_filter_min_count=absa_config["sentence_filter_min_count"],
                       polarity_filter_min_ratio=absa_config["polarity_filter_min_ratio"],
                       aspect_opinion_filter_min_count=absa_config["aspect_opinion_filter_min_count"],
                       aspect_opinion_num_samples=absa_config["aspect_opinion_num_samples"],
                       max_iterations=absa_config["max_iterations"])

    save_aspect_merge_pdf(aspect_candidates_filepath,
                          aspect_merge_filepath,
                          unigram_filepath,
                          absa_config["aspect_opinion_num_samples"])

    save_opinion_rank_pdf(opinion_candidates_filepath,
                          opinion_vecs_filepath,
                          opinion_rank_filepath)
