from typing import Optional, Dict
from utils.general_util import split_filepath
from utils.resource_util import zip_repo
from utils.log_util import get_logger
from pyspark import SparkConf
from pyspark.sql import SparkSession, Window
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pathlib import Path
from pprint import pformat
import pandas as pd
import shutil
import csv
import os


def get_spark_session(app_name: str = "spark_app",
                      config_overrides: Dict = {},
                      master_config: Optional[str] = None,
                      log_level: str = "WARN") -> SparkSession:
    default_config = {
        "spark.sql.execution.arrow.pyspark.enabled": "true",
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.kryoserializer.buffer": "512k",
        "spark.kryoserializer.buffer.max": "1024m",
    }
    default_config.update(config_overrides)
    config = SparkConf().setAll(default_config.items())
    spark_session_builder = SparkSession.builder.appName(app_name).config(conf=config)
    if master_config:
        spark_session_builder.master(master_config)
    spark_session = spark_session_builder.getOrCreate()
    spark_session.sparkContext.setLogLevel(log_level)
    return spark_session


def convert_to_pdf_and_save(spark_df: DataFrame,
                            save_filepath: Optional[str] = None,
                            rename_columns: Optional[Dict] = None,
                            csv_index: bool = False,
                            csv_index_label: Optional[bool] = None,
                            csv_quoting: int = csv.QUOTE_MINIMAL) -> pd.DataFrame:
    pandas_df = spark_df.toPandas()
    if rename_columns is not None:
        pandas_df.columns = [rename_columns.get(i, i) for i in pandas_df.columns]
    file_format = Path(save_filepath).suffix[1:] if save_filepath else None
    if file_format == "csv":
        pandas_df.to_csv(save_filepath, index=csv_index, index_label=csv_index_label, quoting=csv_quoting)
    elif file_format == "json":
        pandas_df.to_json(save_filepath, orient="records", lines=True, force_ascii=False)
    elif file_format is not None:
        raise ValueError(f"Unsupported file format of {file_format}")
    return pandas_df


def write_dataframe_to_dir(dataframe: DataFrame,
                           save_folder_dir: str,
                           save_folder_name: str,
                           file_format: str,
                           num_partitions: Optional[int] = None):
    if num_partitions is not None:
        dataframe = dataframe.coalesce(num_partitions)
    save_directory = os.path.join(save_folder_dir, save_folder_name)
    if file_format == "orc":
        dataframe.write.orc(save_directory)
    elif file_format == "csv":
        dataframe.write.csv(save_directory, header=True, escape='"')
    elif file_format == "json":
        dataframe.write.json(save_directory)
    elif file_format == "txt" or file_format == "text":
        dataframe.write.text(save_directory)
    else:
        raise ValueError(f"Unsupported file format of {file_format}")


def write_dataframe_to_file(dataframe: DataFrame, save_filepath: str):
    file_dir, file_name, file_format = split_filepath(save_filepath)
    write_dataframe_to_dir(dataframe, file_dir, file_name, file_format, num_partitions=1)
    spark_data_dir = os.path.join(file_dir, file_name)
    spark_filename = [i for i in os.listdir(spark_data_dir) if i.startswith("part-")][0]
    spark_datat_dir = os.path.join(file_dir, file_name)
    spark_filepath = os.path.join(spark_datat_dir, spark_filename)
    shutil.move(spark_filepath, save_filepath)
    shutil.rmtree(spark_datat_dir)


def convert_to_orc(spark: DataFrame,
                   input_filepath: str,
                   output_filepath: str,
                   infer_schema: bool = True,
                   type_casting: Optional[dict] = None):
    file_format = Path(input_filepath).suffix[1:]
    if file_format == "csv":
        data_df = spark.read.csv(input_filepath, header=True, quote='"', escape='"', inferSchema=infer_schema)
    elif file_format == "json":
        data_df = spark.read.json(input_filepath)
    else:
        Exception(f"Unsupported file format of {file_format}")

    if type_casting:
        for col, cast_type in type_casting.items():
            data_df = data_df.withColumn(col, F.col(col).cast(cast_type))
    logger = get_logger()
    logger.info(f"data types of orc file:\n{pformat(data_df.dtypes)}")
    write_dataframe_to_file(data_df, output_filepath)


def add_repo_pyfile(spark: SparkSession, repo_zip_dir: str = "/tmp"):
    repo_zip_filepath = zip_repo(repo_zip_dir)
    spark.sparkContext.addPyFile(repo_zip_filepath)


def extract_topn_common(data_df: DataFrame,
                        partition_by: str,
                        key_by: str,
                        value_by: str,
                        top_n: int = 3,
                        save_filepath: Optional[str] = None) -> DataFrame:
    w = Window.partitionBy(partition_by).orderBy(F.col(value_by).desc())
    data_df = data_df.select(partition_by, key_by, value_by)
    data_df = data_df.withColumn("rank", F.row_number().over(w))
    data_df = data_df.filter(F.col("rank") <= top_n).drop("rank")
    data_df = data_df.groupby(partition_by) \
        .agg(F.to_json(F.map_from_entries(F.collect_list(F.struct(key_by, value_by)))).alias(key_by))
    if save_filepath:
        convert_to_pdf_and_save(data_df, save_filepath)
    return data_df
