from annotation.annotation_utils.annotation_util import read_annotation_config
from annotation.components.annotator import pudf_annotate
from utils.spark_util import get_spark_session, write_dataframe_to_dir, add_repo_pyfile
import pyspark.sql.functions as F
from pathlib import Path
import os

if __name__ == "__main__":
    dummy_normalizer_config = {
        "merge_words": {"battery life": {"merge": "batterylife", "type": "canonical"}},
        "split_words": {"autonomouscars": "autonomous cars"},
        "replace_words": {"thisr": "these"},
    }

    annotation_config_filepath = os.path.join(Path(__file__).parent, "annotation.cfg")
    nlp_model_config = read_annotation_config(annotation_config_filepath)
    if nlp_model_config["normalizer_config"]:
        nlp_model_config["normalizer_config"].update(dummy_normalizer_config)

    spark = get_spark_session("small_test", master_config="local[4]", log_level="INFO")
    add_repo_pyfile(spark)


    annotation_dir = "/Users/haiyang/Desktop/annotation/"
    data_filepath = os.path.join(annotation_dir, "medium_test.json")

    data_df = spark.read.text(data_filepath)
    data_df = data_df.repartition(4)
    print(data_df.rdd.getNumPartitions())

    annotation_df = data_df.select(pudf_annotate(F.col("value"), nlp_model_config))
    write_dataframe_to_dir(annotation_df, annotation_dir, "medium_test_annotation", file_format="txt")
