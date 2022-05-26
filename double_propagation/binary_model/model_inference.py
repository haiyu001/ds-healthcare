from double_propagation.binary_model.model_building import get_sentiment_features_pdf, get_model_prediction_pdf
from utils.general_util import setup_logger, save_pdf
from utils.resource_util import get_model_filepath
from word_vector.wv_space import ConceptNetWordVec, load_txt_vecs_to_pdf
import pandas as pd
from scipy.stats import hmean
import os

if __name__ == "__main__":
    setup_logger()

    test_dir = "/Users/haiyang/Desktop/opinion_test"
    conceptnet_vecs_filepath = get_model_filepath("model", "conceptnet", "numberbatch-en-19.08.txt")
    opinion_vecs_filepath = os.path.join(test_dir, "opinion_vecs.txt")
    opinions_filepath = os.path.join(test_dir, "opinions_test.csv")
    sentiment_prediction_filepath = os.path.join(test_dir, "opinions_sentiment_prediction.csv")
    subjectivity_prediction_filepath = os.path.join(test_dir, "opinions_subjectivity_prediction.csv")
    opinion_sentiment_subjectivity_scores_filepath = os.path.join(test_dir, "opinion_sentiment_subjectivity_scores.csv")

    # extract opinion vecs
    opinions = pd.read_csv(opinions_filepath, encoding="utf-8")["text"].tolist()
    wordvec = ConceptNetWordVec(conceptnet_vecs_filepath, use_oov_strategy=True)
    wordvec.extract_txt_vecs(opinions, opinion_vecs_filepath)

    # run sentiment model prediction
    sentiment_features_pdf = get_sentiment_features_pdf(opinion_vecs_filepath)
    sentiment_model_filepath = get_model_filepath("model", "sentiment", "sentiment.hdf5")
    opinion_sentiment_scores_pdf = get_model_prediction_pdf(sentiment_features_pdf,
                                                            sentiment_model_filepath,
                                                            predicted_score_col="sentiment_score",
                                                            save_filepath=sentiment_prediction_filepath)

    # run subjectivity model prediction
    subjectivity_features_pdf = load_txt_vecs_to_pdf(opinion_vecs_filepath)
    subjectivity_model_filepath = get_model_filepath("model", "subjectivity", "subjectivity.hdf5")
    opinion_subjectivity_scores_pdf = get_model_prediction_pdf(subjectivity_features_pdf,
                                                               subjectivity_model_filepath,
                                                               predicted_score_col="subjectivity_score",
                                                               save_filepath=subjectivity_prediction_filepath)

    # save opinion sentiment subjectivity scores
    opinion_sentiment_subjectivity_scores_pdf = \
        opinion_sentiment_scores_pdf.merge(opinion_subjectivity_scores_pdf, on="word")
    opinion_sentiment_subjectivity_scores_pdf["hmean_score"] = hmean(opinion_sentiment_subjectivity_scores_pdf,
                                                                       axis=1)
    save_pdf(opinion_sentiment_subjectivity_scores_pdf, opinion_sentiment_subjectivity_scores_filepath, csv_index=True)
