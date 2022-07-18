from typing import Optional
from pyspark.sql import SparkSession, DataFrame
from utils.general_util import get_filepaths_recursively


def load_annotation(spark: SparkSession,
                    annotation_dir: str,
                    drop_non_english: bool = True,
                    num_partitions: Optional[int] = None) -> DataFrame:
    annotation_filepaths = get_filepaths_recursively(annotation_dir, ["json", "txt"])
    annotation_sdf = spark.read.json(annotation_filepaths)
    if drop_non_english:
        annotation_sdf = annotation_sdf.filter(annotation_sdf["_"]["language"]["lang"] == "en")
    if num_partitions is not None:
        annotation_sdf = annotation_sdf.repartion(num_partitions)
    return annotation_sdf


