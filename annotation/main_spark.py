from annotation.annotation_utils.annotator_util import get_canonicalization_nlp_model_config, get_nlp_model_config
from annotation.components.annotator import pudf_annotate
from utils.config_util import read_config_to_dict
from utils.general_util import setup_logger
from utils.resource_util import get_data_filepath, get_repo_dir
from utils.spark_util import get_spark_session, write_sdf_to_dir, add_repo_pyfile
import pyspark.sql.functions as F
import os


if __name__ == "__main__":
    setup_logger()

    annotation_config_filepath = os.path.join(get_repo_dir(), "annotation", "pipelines", "conf/annotation_template.cfg")
    annotation_config = read_config_to_dict(annotation_config_filepath)
    nlp_model_config_filepath = os.path.join(get_repo_dir(), "annotation", "pipelines", "conf/nlp_model_template.cfg")
    domain_dir = get_data_filepath(annotation_config["domain"])
    canonicalization_dir = os.path.join(domain_dir, annotation_config["canonicalization_folder"])

    input_file_name = "small_drug_reviews.json"
    spark_cores = 2

    # config_updates = {"spark.archives": "/Users/haiyang/github/datascience.tar.gz"}
    spark = get_spark_session("test", master_config=f"local[{spark_cores}]", log_level="INFO")
    add_repo_pyfile(spark)
    input_filepath = os.path.join(domain_dir, "input", input_file_name)

    # ======================================== canonicalizer =========================================

    input_sdf = spark.read.text(input_filepath).repartition(spark_cores)
    canonicalization_nlp_model_config = get_canonicalization_nlp_model_config(nlp_model_config_filepath)
    canonicalization_annotation_sdf = input_sdf.select(pudf_annotate(F.col("value"), canonicalization_nlp_model_config))
    write_sdf_to_dir(canonicalization_annotation_sdf, canonicalization_dir,
                     annotation_config["canonicalization_annotation_folder"], file_format="txt")

    # ======================================== annotator ===============================================

    input_sdf = spark.read.text(input_filepath).repartition(spark_cores)
    normalization_json_filepath = os.path.join(canonicalization_dir, "normalization.json")
    nlp_model_config = get_nlp_model_config(nlp_model_config_filepath, normalization_json_filepath)
    annotation_sdf = input_sdf.select(pudf_annotate(F.col("value"), nlp_model_config))
    write_sdf_to_dir(annotation_sdf, domain_dir, annotation_config["annotation_folder"], file_format="txt")
