from typing import Dict, List, Optional
from ml.binary_mlp import BinaryModel
from ml.ml_utils import get_train_val_test
from utils.general_util import setup_logger, save_pdf, load_json_file, make_dir
from utils.resource_util import get_model_filepath
from word_vector.wv_space import WordVec, load_txt_vecs_to_pdf
import pandas as pd
import tensorflow as tf
import os


def get_similarity_stats_pdf(similarity_matrix_pdf: pd.DataFrame, prefix: str):
    stats_pdf = pd.DataFrame({f"{prefix}_min": similarity_matrix_pdf.min(axis=1),
                              f"{prefix}_max": similarity_matrix_pdf.max(axis=1),
                              f"{prefix}_std": similarity_matrix_pdf.std(axis=1),
                              f"{prefix}_avg": similarity_matrix_pdf.mean(axis=1)})
    return stats_pdf


def get_sentiment_features_pdf(sentiment_vecs_filepath: str) -> pd.DataFrame:
    sentiment_model_dir = get_model_filepath("model", "sentiment", "training")
    opinion_neg = WordVec(os.path.join(sentiment_model_dir, "absa_seed_opinions_neg_vecs.txt"))
    opinion_pos = WordVec(os.path.join(sentiment_model_dir, "absa_seed_opinions_pos_vecs.txt"))
    sentiment_vecs = WordVec(sentiment_vecs_filepath)
    sentiment_neg_similarity_pdf = sentiment_vecs.norm_vecs_pdf.dot(opinion_neg.norm_vecs_pdf.T)
    sentiment_pos_similarity_pdf = sentiment_vecs.norm_vecs_pdf.dot(opinion_pos.norm_vecs_pdf.T)
    neg_stats_features_pdf = get_similarity_stats_pdf(sentiment_neg_similarity_pdf, "neg")
    pos_stats_features_pdf = get_similarity_stats_pdf(sentiment_pos_similarity_pdf, "pos")
    sentiment_vecs_features_pdf = load_txt_vecs_to_pdf(sentiment_vecs_filepath)
    sentiment_features_pdf = pd.concat(
        [neg_stats_features_pdf, pos_stats_features_pdf, sentiment_vecs_features_pdf], axis=1)
    return sentiment_features_pdf


def get_sentiment_training_pdf(sentiment_vecs_filepath: str, sentiment_labels: Dict[str, int]):
    sentiment_features_pdf = get_sentiment_features_pdf(sentiment_vecs_filepath)
    sentiment_features_pdf["label"] = [sentiment_labels[word] for word in sentiment_features_pdf.index]
    sentiment_training_pdf = sentiment_features_pdf[
        ["label"] + [i for i in sentiment_features_pdf.columns if i != "label"]]
    non_sentiment_cnt = sentiment_training_pdf[sentiment_training_pdf["label"] == 0].shape[0]
    sentiment_cnt = sentiment_training_pdf[sentiment_training_pdf["label"] == 1].shape[0]
    sentiment_training_filepath = get_model_filepath(
        "model", "sentiment", "training", f"sentiment_no-{non_sentiment_cnt}_yes-{sentiment_cnt}_training.csv")
    save_pdf(sentiment_training_pdf, sentiment_training_filepath, csv_index_label="word", csv_index=True)


def get_concreteness_training_pdf():
    concreteness_filepath = "/Users/haiyang/Desktop/concreteness.json"
    concreteness_dict = load_json_file(concreteness_filepath)
    conceptnet_vecs_filepath = get_model_filepath("model", "conceptnet", "numberbatch-en-19.08.txt")
    conceptnet_vecs = WordVec(conceptnet_vecs_filepath, use_oov_strategy=False)
    conceptnet_vocab = set(conceptnet_vecs.get_vocab())
    concreteness_dict = {k: v for k, v in concreteness_dict.items() if k in conceptnet_vocab}
    concreteness_words = list(concreteness_dict.keys())
    concreteness_vecs_txt_filepath = "/Users/haiyang/Desktop/concreteness_vecs.txt"
    conceptnet_vecs.extract_txt_vecs(concreteness_words, concreteness_vecs_txt_filepath, l2_norm=False)
    concreteness_vecs_pdf = load_txt_vecs_to_pdf(concreteness_vecs_txt_filepath)
    concreteness_vecs_pdf["label"] = list(concreteness_dict.values())
    concreteness_traning_pdf = concreteness_vecs_pdf[
        ["label"] + [i for i in concreteness_vecs_pdf.columns if i != "label"]]
    abstractness_cnt = concreteness_traning_pdf[concreteness_traning_pdf["label"] == 0].shape[0]
    concreteness_cnt = concreteness_traning_pdf[concreteness_traning_pdf["label"] == 1].shape[0]
    concreteness_training_filepath = get_model_filepath(
        "model", "concreteness", "training", f"concreteness_no-{abstractness_cnt}_yes-{concreteness_cnt}_training_tmp.csv")
    save_pdf(concreteness_traning_pdf, concreteness_training_filepath, csv_index_label="word", csv_index=True)


def get_model_prediction_pdf(features_pdf: List[str],
                             model_filepath: str,
                             predicted_score_col: str = "predicted_score",
                             save_filepath: Optional[str] = None) -> pd.DataFrame:
    model = tf.keras.models.load_model(model_filepath)
    predictions = model.predict(x=features_pdf, verbose=1)
    prediction_pdf = pd.DataFrame({predicted_score_col: predictions[:, 0]}, index=features_pdf.index)
    if save_filepath:
        save_pdf(prediction_pdf, save_filepath, csv_index=True)
    return prediction_pdf


if __name__ == "__main__":
    setup_logger()

    # ============================================ sentiment model ===============================================
    #
    # sentiment_vecs_filepath = get_model_filepath("model", "sentiment", "training", "sentiment_vecs.txt")
    # sentiment_labels = load_json_file(get_model_filepath("model", "sentiment", "training", "sentiment_labels.json"))
    # get_sentiment_training_pdf(sentiment_vecs_filepath, sentiment_labels)
    #
    # sentiment_model_dir = get_model_filepath("model", "sentiment", "training")
    # sentiment_model_training_data_filepath = os.path.join(sentiment_model_dir,
    #                                                       "sentiment_no-3573_yes-3566_training.csv")
    # features_pdf = pd.read_csv(sentiment_model_training_data_filepath, index_col="word", encoding="utf-8")
    #
    # for i in range(5, 9):
    #     print("=" * 50, i, "=" * 50)
    #     features_pdf = features_pdf.sample(frac=1.0)
    #     sentiment_i_dir = os.path.join(sentiment_model_dir, f"sentiment_{i}")
    #     model = BinaryModel(make_dir(sentiment_i_dir),
    #                         input_dimension=308,
    #                         class_names=["non-sentiment", "sentiment"],
    #                         learning_rate=0.0001,
    #                         dropout_rate=0.5,
    #                         first_hidden_layer_size=64,
    #                         second_hidden_layer_size=16,
    #                         epochs=200,
    #                         batch_size=16)
    #     X_train, y_train, X_val, y_val, X_test, y_test = get_train_val_test(features_pdf, val_size=0.05, test_size=0.05)
    #     model.train_model(X_train, y_train, X_val, y_val, X_test, y_test, class_weight=None)

    # # ============================================== subjectivity model ==============================================
    #
    # subjectivity_model_dir = get_model_filepath("model", "subjectivity", "training")
    # subjectivity_model_training_data_filepath = os.path.join(subjectivity_model_dir,
    #                                                          "subjectivity_weak-2698_strong-4515_training.csv")
    # features_pdf = pd.read_csv(subjectivity_model_training_data_filepath, index_col="word", encoding="utf-8")
    #
    # for i in range(2, 6):
    #     print("=" * 50, i, "=" * 50)
    #     features_pdf = features_pdf.sample(frac=1.0)
    #     subjectivity_i_dir = os.path.join(subjectivity_model_dir, f"subjectivity_{i}")
    #     model = BinaryModel(make_dir(subjectivity_i_dir),
    #                         input_dimension=300,
    #                         class_names=["weaksubj", "strongsubj"],
    #                         learning_rate=0.0001,
    #                         dropout_rate=0.4,
    #                         first_hidden_layer_size=64,
    #                         second_hidden_layer_size=32,
    #                         epochs=300,
    #                         batch_size=16)
    #     X_train, y_train, X_val, y_val, X_test, y_test = get_train_val_test(features_pdf, val_size=0.05, test_size=0.05)
    #     model.train_model(X_train, y_train, X_val, y_val, X_test, y_test, class_weight={0: 0.6, 1: 0.4})

    # # ============================================== subjectivity model ==============================================

    get_concreteness_training_pdf()

    # concreteness_model_dir = get_model_filepath("model", "concreteness", "training")
    # concreteness_model_training_data_filepath = os.path.join(concreteness_model_dir,
    #                                                          "concreteness_no-4766_yes-6058_training.csv")
    # features_pdf = pd.read_csv(concreteness_model_training_data_filepath, index_col="word", encoding="utf-8")
    #
    # for i in range(1, 6):
    #     print("=" * 50, i, "=" * 50)
    #     features_pdf = features_pdf.sample(frac=1.0)
    #     subjectivity_i_dir = os.path.join(concreteness_model_dir, f"concreteness_{i}")
    #     model = BinaryModel(make_dir(subjectivity_i_dir),
    #                         input_dimension=300,
    #                         class_names=["abstractness", "concreteness"],
    #                         learning_rate=0.0001,
    #                         dropout_rate=0.5,
    #                         first_hidden_layer_size=64,
    #                         second_hidden_layer_size=32,
    #                         epochs=200,
    #                         batch_size=16)
    #     X_train, y_train, X_val, y_val, X_test, y_test = get_train_val_test(features_pdf, val_size=0.05, test_size=0.05)
    #     model.train_model(X_train, y_train, X_val, y_val, X_test, y_test, class_weight={0: 0.6, 1: 0.4})


