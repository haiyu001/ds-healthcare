from typing import Dict, Any, Iterator, Optional
from annotation.components.annotator import get_nlp_model, doc_to_json_str
from utils.general_util import get_filepaths_recursively
from pyspark.sql import Column, SparkSession, DataFrame
from pyspark.sql.types import StringType
import pyspark.sql.functions as F
import pandas as pd


def pudf_annotate(text_iter: Column, nlp_model_config: Dict[str, Any]) -> Column:
    def annotate(text_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
        nlp = get_nlp_model(**nlp_model_config)
        for text in text_iter:
            doc = text.apply(nlp)
            doc_json_str = doc.apply(doc_to_json_str)
            yield doc_json_str

    return F.pandas_udf(annotate, StringType())(text_iter)


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