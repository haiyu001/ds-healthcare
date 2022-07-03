from utils.resource_util import get_model_filepath


def get_mallet_model_filepath() -> str:
    mallet_model_filepath = get_model_filepath("model", "Mallet-202108", "bin", "mallet")
    return mallet_model_filepath