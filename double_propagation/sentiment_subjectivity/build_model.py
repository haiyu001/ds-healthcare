from ml.binary_mlp import BinaryModel
from ml.ml_utils import get_train_val_test
from utils.general_util import setup_logger, save_pdf, load_json_file, make_dir
from utils.resource_util import get_model_filepath
from word_vector.wv_space import WordVec, load_txt_vecs_to_pdf
import tensorflow as tf
import pandas as pd
import os


def get_similarity_stats_pdf(sim_df, prefix):
    stats_pdf = pd.DataFrame({f"{prefix}_min": sim_df.min(axis=1),
                              f"{prefix}_max": sim_df.max(axis=1),
                              f"{prefix}_std": sim_df.std(axis=1),
                              f"{prefix}_avg": sim_df.mean(axis=1)})
    return stats_pdf


def get_sentiment_features_df(sentiment_neg_similarity_pdf, sentiment_pos_similarity_pdf, sentiment_features_pdf):
    neg_stats_pdf = get_similarity_stats_pdf(sentiment_neg_similarity_pdf, "neg")
    pos_stats_pdf = get_similarity_stats_pdf(sentiment_pos_similarity_pdf, "pos")
    features_pdf = pd.concat([neg_stats_pdf, pos_stats_pdf, sentiment_features_pdf], axis=1)
    features_pdf = features_pdf[["label"] + [i for i in features_pdf.columns if i != "label"]]
    non_sentiment_cnt = features_pdf[features_pdf["label"] == 0].shape[0]
    sentiment_cnt = features_pdf[features_pdf["label"] == 1].shape[0]
    save_pdf(
        features_pdf,
        get_model_filepath("model", "sentiment", f"sentiment_no-{non_sentiment_cnt}_yes-{sentiment_cnt}_training.csv"),
        csv_index_label="word", csv_index=True
    )
    return features_pdf


if __name__ == "__main__":
    setup_logger()

    # ============================================ sentiment features =============================================

    # sentiment_features_df = load_txt_vecs_to_pdf(get_model_filepath("model", "sentiment", "sentiment_vecs.txt"))
    # sentiment_labels = load_json_file(get_model_filepath("model", "sentiment", "sentiment_labels.json"))
    # sentiment_features_df["label"] = [sentiment_labels[word] for word in sentiment_features_df.index]
    #
    # sentiment_vec = WordVec(get_model_filepath("model", "sentiment", "sentiment_vecs.txt"))
    # opinion_neg = WordVec(get_model_filepath("model", "sentiment", "absa_seed_opinions_neg_vecs.txt"))
    # opinion_pos = WordVec(get_model_filepath("model", "sentiment", "absa_seed_opinions_pos_vecs.txt"))
    # sentiment_neg_similarity_pdf = sentiment_vec.vecs_pdf.dot(opinion_neg.vecs_pdf.T)
    # sentiment_pos_similarity_pdf = sentiment_vec.vecs_pdf.dot(opinion_pos.vecs_pdf.T)
    # get_sentiment_features_df(sentiment_neg_similarity_pdf, sentiment_pos_similarity_pdf, sentiment_features_df)

    # # ============================================== sentiment model ==============================================

    sentiment_model_dir = get_model_filepath("model", "sentiment")
    sentiment_model_training_data_filepath = os.path.join(sentiment_model_dir, "sentiment_no-3573_yes-3566_training.csv")
    features_df = pd.read_csv(sentiment_model_training_data_filepath, index_col="word", encoding="utf-8")

    for i in range(5, 9):
        print("=" * 50, i, "=" * 50)
        features_df = features_df.sample(frac=1.0)
        sentiment_i_dir = os.path.join(sentiment_model_dir, f"sentiment_{i}")
        model = BinaryModel(make_dir(sentiment_i_dir),
                            input_dimension=308,
                            class_names=["non-sentiment", "sentiment"],
                            learning_rate=0.0001,
                            dropout_rate=0.5,
                            first_hidden_layer_size=64,
                            second_hidden_layer_size=16,
                            epochs=200,
                            batch_size=16)
        X_train, y_train, X_val, y_val, X_test, y_test = get_train_val_test(features_df, val_size=0.05, test_size=0.05)
        model.train_model(X_train, y_train, X_val, y_val, X_test, y_test, class_weight=None)

    # # ============================================== subjectivity model ==============================================
    #
    # subjectivity_model_dir = get_model_filepath("model", "subjectivity")
    # subjectivity_model_training_data_filepath = os.path.join(subjectivity_model_dir,
    #                                                          "subjectivity_weak-2698_strong-4515_training.csv")
    # features_df = pd.read_csv(subjectivity_model_training_data_filepath, index_col="word", encoding="utf-8")
    #
    # for i in range(2, 6):
    #     print("=" * 50, i, "=" * 50)
    #     features_df = features_df.sample(frac=1.0)
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
    #     X_train, y_train, X_val, y_val, X_test, y_test = get_train_val_test(features_df, val_size=0.05, test_size=0.05)
    #     model.train_model(X_train, y_train, X_val, y_val, X_test, y_test, class_weight={0: 0.6, 1: 0.4})