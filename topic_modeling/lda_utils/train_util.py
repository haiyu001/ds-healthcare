from utils.resource_util import get_model_filepath


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

