import pyspark.sql.functions as F
from pyspark.sql import DataFrame


def get_canonicalization_candidates(vocab_sdf: DataFrame,
                                    bigram_sdf: DataFrame,
                                    canonicalization_candidates_filepath: str,
                                    num_partitions: int = 1):
    vocab_sdf = vocab_sdf.select(F.col("word"),
                                 F.col("count").alias("word_count"),
                                 F.col("top_three_pos"))
    bigram_sdf = bigram_sdf.select(F.regexp_replace(F.col("ngram"), " ", "").alias("word"),
                                   F.col("ngram").alias("bigram"),
                                   F.col("count").alias("bigram_count"))
    canonicalization_candidates_sdf = vocab_sdf.join(bigram_sdf, on="word", how="inner")
    canonicalization_candidates_sdf = canonicalization_candidates_sdf\
        .select("word", "bigram", "word_count", "bigram_count", "top_three_pos")
    write_sdf_to_file(canonicalization_candidates_sdf, canonicalization_candidates_filepath, num_partitions)


if __name__ == "__main__":
    from annotation.annotation_utils.annotation_util import read_annotation_config
    from utils.resource_util import get_data_filepath, get_repo_dir
    from utils.spark_util import get_spark_session, write_sdf_to_file
    import os

    annotation_config_filepath = os.path.join(get_repo_dir(), "conf", "annotation_template.cfg")
    annotation_config = read_annotation_config(annotation_config_filepath)

    domain_dir = get_data_filepath(annotation_config["domain"])
    extraction_folder = annotation_config["extraction_folder"]
    canonicalization_folder = annotation_config["canonicalization_folder"]

    spark = get_spark_session("test", master_config="local[4]", log_level="WARN")

    vocab_filepath = os.path.join(domain_dir, extraction_folder, annotation_config["vocab_filename"])
    bigram_filepath = os.path.join(domain_dir, extraction_folder, annotation_config["bigram_filename"])
    canonicalization_candidates_filepath = os.path.join(domain_dir, canonicalization_folder,
                                                        annotation_config["canonicalization_candidates_filename"])

    vocab_sdf = spark.read.csv(vocab_filepath, header=True, quote='"', escape='"', inferSchema=True)
    bigram_sdf = spark.read.csv(bigram_filepath, header=True, quote='"', escape='"', inferSchema=True)
    get_canonicalization_candidates(vocab_sdf, bigram_sdf, canonicalization_candidates_filepath)