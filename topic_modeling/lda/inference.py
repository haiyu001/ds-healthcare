from typing import List, Tuple, Dict
from pyspark.sql import SparkSession, Column
from pyspark.sql.types import StringType
from topic_modeling.lda.mallet_wrapper import LdaMallet
from utils.general_util import load_pickle_file, save_pdf
from pyspark.sql import functions as F
from utils.spark_util import write_sdf_to_dir
import json


def udf_filter_by_threshold(topics: Column,
                            vis_topic_id_to_org_topics: Dict[str, List[int]],
                            inference_threshold) -> Column:
    def filter_by_threshold(topics):
        if topics.startswith("#doc"):
            return None
        parts = topics.split()
        topic_probs = [float(prob) for prob in parts[2:]]
        doc_topics = {"doc_id": parts[1],
                      "topics": dict()}
        for topic_id, org_topics in vis_topic_id_to_org_topics.items():
            topic_prob = sum([topic_probs[id] for id in org_topics])
            if topic_prob >= inference_threshold:
                doc_topics["topics"][topic_id] = topic_prob
        doc_topics_str = json.dumps(doc_topics, ensure_ascii=False)
        return doc_topics_str
    return F.udf(filter_by_threshold, StringType())(topics)


def extract_doc_topics(mallet_corpus_filepath: str,
                       mallet_model_filepath: str) -> List[List[Tuple[int, float]]]:
    mallet_corpus: List[Tuple[str, List[Tuple[int, int]]]] = load_pickle_file(mallet_corpus_filepath)
    mallet_model = LdaMallet.load(mallet_model_filepath)
    return mallet_model[mallet_corpus]


def predict_topics(spark: SparkSession,
                   doc_topics_infer_filepath: str,
                   vis_topic_id_to_org_topics: Dict[str, List[int]],
                   inference_threshold: float,
                   save_folder_dir: str,
                   save_folder_name: str):
    doc_topics_pdf = spark.read.text(doc_topics_infer_filepath)
    doc_topics_pdf = doc_topics_pdf.select(udf_filter_by_threshold(F.col("value"),
                                                                   vis_topic_id_to_org_topics,
                                                                   inference_threshold))
    doc_topics_pdf = doc_topics_pdf.dropna()
    write_sdf_to_dir(doc_topics_pdf, save_folder_dir, save_folder_name, "txt")


def extract_topics_stats(spark: SparkSession,
                         lda_inference_dir: str,
                         lda_stats_filepath: str):
    lda_inference_sdf = spark.read.json(lda_inference_dir)
    num_docs = lda_inference_sdf.count()
    lda_inference_sdf = lda_inference_sdf.select(F.col("topics.*"))
    sorted_topic_ids = [str(i) for i in sorted([int(i) for i in lda_inference_sdf.columns])]
    lda_stats_sdf = lda_inference_sdf.select(
        [F.count(F.when(F.col(c).isNotNull(), c)).alias(c) for c in sorted_topic_ids])
    lda_stats_pdf = lda_stats_sdf.toPandas().T
    lda_stats_pdf = lda_stats_pdf.rename(columns={0: "count"})
    lda_stats_pdf["percent"] = round((lda_stats_pdf["count"] / num_docs) * 100, 2)
    save_pdf(lda_stats_pdf, lda_stats_filepath, csv_index=True, csv_index_label="topic_id")
