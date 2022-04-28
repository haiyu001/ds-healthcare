from typing import Optional, Dict, List
from utils.general_util import split_filepath, save_pdf
from utils.resource_util import zip_repo
from pyspark import SparkConf
from pyspark.sql import SparkSession, Window
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pathlib import Path
from pprint import pformat
import pandas as pd
import logging
import shutil
import os


def get_spark_session(app_name: str = "spark_app",
                      config_updates: Dict = {},
                      master_config: Optional[str] = None,
                      log_level: str = "WARN") -> SparkSession:
    default_config = {
        "spark.sql.execution.arrow.pyspark.enabled": "true",
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.kryoserializer.buffer": "512k",
        "spark.kryoserializer.buffer.max": "1024m",
    }
    default_config.update(config_updates)
    config = SparkConf().setAll(default_config.items())
    spark_session_builder = SparkSession.builder.appName(app_name).config(conf=config)
    if master_config:
        spark_session_builder.master(master_config)
    spark_session = spark_session_builder.getOrCreate()
    spark_session.sparkContext.setLogLevel(log_level)
    return spark_session


def write_sdf_to_dir(sdf: DataFrame,
                     save_folder_dir: str,
                     save_folder_name: str,
                     file_format: str,
                     num_partitions: Optional[int] = None):
    if num_partitions is not None:
        sdf = sdf.coalesce(num_partitions)
    save_directory = os.path.join(save_folder_dir, save_folder_name)
    if file_format == "orc":
        sdf.write.orc(save_directory)
    elif file_format == "csv":
        sdf.write.csv(save_directory, header=True, escape='"')
    elif file_format == "json":
        sdf.write.json(save_directory)
    elif file_format == "txt" or file_format == "text":
        sdf.write.text(save_directory)
    else:
        raise ValueError(f"Unsupported file format of {file_format}")


def _merge_spark_text_files(save_filepath: str, part_filepaths: List[str]):
    with open(save_filepath, "w", encoding="utf-8") as output_file:
        for part_filepath in part_filepaths:
            with open(part_filepath, "r", encoding="utf-8") as input_file:
                for line in input_file:
                    output_file.write(line)


def write_sdf_to_file(sdf: DataFrame, save_filepath: str, num_partitions: Optional[int] = None):
    file_dir, file_name, file_format = split_filepath(save_filepath)
    write_sdf_to_dir(sdf, file_dir, file_name, file_format, num_partitions=num_partitions)
    spark_data_dir = os.path.join(file_dir, file_name)
    part_filepaths = [os.path.join(spark_data_dir, part_filename)
                      for part_filename in os.listdir(spark_data_dir) if part_filename.startswith("part-")]
    if len(part_filepaths) > 1:
        if file_format == "txt" or file_format == "text":
            _merge_spark_text_files(save_filepath, part_filepaths)
        else:
            pdf_list = []
            for part_filepath in part_filepaths:
                if file_format == "csv":
                    pdf = pd.read_csv(part_filepath, keep_default_na=False, na_values="", encoding="utf-8")
                elif file_format == "json":
                    pdf = pd.read_json(part_filepath, orient="records", lines=True, encoding="utf-8")
                else:
                    ValueError(f"Unsupported file format of {file_format} when num_partitions > 1")
                pdf_list.append(pdf)
            pdf = pd.concat(pdf_list, ignore_index=True)
            save_pdf(pdf, save_filepath)
    else:
        shutil.move(part_filepaths[0], save_filepath)
    shutil.rmtree(spark_data_dir)


def convert_to_orc(spark: SparkSession,
                   input_filepath: str,
                   output_filepath: str,
                   infer_schema: bool = True,
                   type_casting: Optional[dict] = None):
    file_format = Path(input_filepath).suffix[1:]
    if file_format == "csv":
        sdf = spark.read.csv(input_filepath, header=True, quote='"', escape='"', inferSchema=infer_schema)
    elif file_format == "json":
        sdf = spark.read.json(input_filepath)
    else:
        Exception(f"Unsupported file format of {file_format}")

    if type_casting:
        for col, cast_type in type_casting.items():
            sdf = sdf.withColumn(col, F.col(col).cast(cast_type))
    logging.info(f"data types of orc file:\n{pformat(sdf.dtypes)}")
    write_sdf_to_file(sdf, output_filepath)


def add_repo_pyfile(spark: SparkSession, repo_zip_dir: str = "/tmp"):
    repo_zip_filepath = zip_repo(repo_zip_dir)
    spark.sparkContext.addPyFile(repo_zip_filepath)


def extract_topn_common(sdf: DataFrame,
                        partition_by: str,
                        key_by: str,
                        value_by: str,
                        topn: int = 3) -> DataFrame:
    w = Window.partitionBy(partition_by).orderBy(F.col(value_by).desc())
    sdf = sdf.select(partition_by, key_by, value_by)
    sdf = sdf.withColumn("rank", F.row_number().over(w))
    sdf = sdf.filter(F.col("rank") <= topn).drop("rank")
    sdf = sdf.groupby(partition_by) \
        .agg(F.to_json(F.map_from_entries(F.collect_list(F.struct(key_by, value_by)))).alias(key_by))
    return sdf


def union_sdfs(*sdfs: DataFrame) -> DataFrame:
    all_sdf = sdfs[0]
    for sdf in sdfs[1:]:
        all_sdf = all_sdf.unionByName(sdf, allowMissingColumns=True)
    return all_sdf

