from pyspark.pandas import DataFrame


def extract_triplet(annotation_sdf: DataFrame):



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
    aspect_grouping_filepath = os.path.join(absa_aspect_dir, absa_config["aspect_grouping_filename"])
    opinion_grouping_filepath = os.path.join(absa_opinion_dir, absa_config["opinion_grouping_filename"])

    spark_cores = 4
    spark = get_spark_session("test", master_config=f"local[{spark_cores}]", log_level="Warn")

    annotation_sdf = load_annotation(spark, annotation_dir, absa_config["drop_non_english"])