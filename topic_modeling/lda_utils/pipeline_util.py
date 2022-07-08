from utils.resource_util import get_model_filepath
import os


def get_mallet_filepath() -> str:
    mallet_filepath = get_model_filepath("model", "Mallet-202108", "bin", "mallet")
    return mallet_filepath


def get_model_folder_name(iterations: int,
                          optimize_interval: int,
                          topic_alpha: float,
                          num_topics: int) -> str:
    model_folder_name = f"mallet_iterations-{iterations}_" \
                        f"optimize_interval-{optimize_interval}_" \
                        f"topic_alpha-{topic_alpha}_" \
                        f"nun_topics-{num_topics}_lda"
    return model_folder_name


def get_model_filename(iterations: int,
                       optimize_interval: int,
                       topic_alpha: float,
                       num_topics: int) -> str:
    model_filename = f"mallet_{iterations}-{optimize_interval}-{topic_alpha}-{num_topics}_lda"
    return model_filename


def get_prefix_by_mallet_model_filepath(mallet_model_filepath: str) -> str:
    model_dir = os.path.dirname(mallet_model_filepath)
    mallet_model_filename = os.path.basename(mallet_model_filepath)
    prefix = os.path.join(model_dir, "tmp_" + mallet_model_filename + "_")
    return prefix


def get_finetune_model_filepath(finetune_model_dir: str,
                                iterations: int,
                                optimize_interval: int,
                                topic_alpha: float,
                                num_topics: int):
    model_folder_name = get_model_folder_name(iterations, optimize_interval, topic_alpha, num_topics)
    dest_model_dir = os.path.join(finetune_model_dir, model_folder_name)
    mallet_model_filename = get_model_filename(iterations, optimize_interval, topic_alpha, num_topics)
    finetune_model_filepath = os.path.join(dest_model_dir, mallet_model_filename)
    return finetune_model_filepath
