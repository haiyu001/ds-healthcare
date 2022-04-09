from typing import Optional, Dict
from utils.general_util import split_filepath, save_pandas_dataframe
from utils.resource_util import zip_repo
from utils.log_util import get_logger
from pyspark.sql.types import StringType
from pyspark import SparkConf
from pyspark.sql import SparkSession, Window, Column
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from collections import Counter
from pathlib import Path
from pprint import pformat
import pandas as pd
import shutil
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


def write_dataframe_to_file(dataframe: DataFrame, save_filepath: str, num_partitions: int = 1):
    file_dir, file_name, file_format = split_filepath(save_filepath)
    if file_format not in ("csv", "json"):
        raise ValueError(f"Unsupported file format of {file_format}")
    write_dataframe_to_dir(dataframe, file_dir, file_name, file_format, num_partitions=num_partitions)
    spark_data_dir = os.path.join(file_dir, file_name)
    part_filepaths = [os.path.join(spark_data_dir, part_filename)
                      for part_filename in os.listdir(spark_data_dir) if part_filename.startswith("part-")]
    if num_partitions > 1:
        pandas_df_list = []
        for part_filepath in part_filepaths:
            pandas_df = pd.read_csv(part_filepath, encoding="utf-8") if file_format == "csv" \
                else pd.read_json(part_filepath, orient="records", lines=True, encoding="utf-8")
            pandas_df_list.append(pandas_df)
        pandas_df = pd.concat(pandas_df_list, ignore_index=True)
        save_pandas_dataframe(pandas_df, save_filepath)
    else:
        shutil.move(part_filepaths[0], save_filepath)
    shutil.rmtree(spark_data_dir)


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
        write_dataframe_to_file(data_df, save_filepath)
    return data_df


def pudf_get_most_common_text(texts: Column):
    def pudf_get_most_common_text(texts: pd.Series) -> pd.Series:
        most_common_text = texts.apply(lambda x: Counter(x).most_common(1)[0][0])
        return most_common_text

    return F.pandas_udf(pudf_get_most_common_text, StringType())(texts)
