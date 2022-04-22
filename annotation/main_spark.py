from annotation.annotation_utils.annotation_util import read_annotation_config, get_canonicalization_nlp_model_config, \
    get_full_nlp_model_config
from annotation.components.annotator import pudf_annotate
from utils.resource_util import get_data_filepath, get_repo_dir
from utils.spark_util import get_spark_session, write_sdf_to_dir, add_repo_pyfile
import pyspark.sql.functions as F
from pprint import pprint
import os

if __name__ == "__main__":
    annotation_config_filepath = os.path.join(get_repo_dir(), "conf", "annotation_template.cfg")
    annotation_config = read_annotation_config(annotation_config_filepath)
    nlp_model_config_filepath = os.path.join(get_repo_dir(), "conf", "nlp_model_template.cfg")
    domain_dir = get_data_filepath(annotation_config["domain"])
    canonicalization_dir = os.path.join(domain_dir, annotation_config["canonicalization_folder"])
    normalization_filepath = os.path.join(canonicalization_dir, annotation_config["normalization_filename"])
    input_filepath = os.path.join(domain_dir, "input", "drug_reviews.json")

    # config_updates = {"spark.archives": "/Users/haiyang/github/datascience.tar.gz"}
    spark_cores = 2
    spark = get_spark_session("test", master_config=f"local[{spark_cores}]", log_level="INFO")
    add_repo_pyfile(spark)

    # canonicalization annotation
    input_sdf = spark.read.text(input_filepath).repartition(spark_cores)
    canonicalization_nlp_model_config = get_canonicalization_nlp_model_config(nlp_model_config_filepath)
    pprint(canonicalization_nlp_model_config)
    canonicalization_annotation_sdf = input_sdf.select(pudf_annotate(F.col("value"), canonicalization_nlp_model_config))
    write_sdf_to_dir(canonicalization_annotation_sdf, canonicalization_dir,
                     annotation_config["canonicalization_annotation_folder"], file_format="txt")

    # # full annotation
    # input_sdf = spark.read.text(input_filepath).repartition(spark_cores)
    # full_nlp_model_config = get_full_nlp_model_config(nlp_model_config_filepath, normalization_filepath)
    # full_annotation_sdf = input_sdf.select(pudf_annotate(F.col("value"), full_nlp_model_config))
    # write_sdf_to_dir(full_annotation_sdf, domain_dir, annotation_config["full_annotation_folder"], file_format="txt")
