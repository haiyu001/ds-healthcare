from topic_modeling.lda.corpus import load_mallet_corpus
from topic_modeling.lda.training import train_mallet_lda_model
from topic_modeling.lda.visualization import save_lda_vis
from utils.config_util import read_config_to_dict
from utils.general_util import make_dir, setup_logger
from utils.resource_util import get_data_filepath
import argparse
import logging
import os


def train_mallet_lda_models(mallet_docs_filepath: str,
                            mallet_id2word_filepath: str,
                            mallet_corpus_filepath: str,
                            mallet_corpus_csc_filepath: str,
                            workers: int,
                            iterations: int,
                            optimize_interval_candidates: str,
                            topic_alpha_candidates: str,
                            num_topics_candidates: str,
                            build_lda_vis: bool,
                            models_dir: str,
                            models_coherence_filepath: str):
    logging.info(f"\n{'*' * 150}\n* build mallet LDA models\n{'*' * 150}\n")
    mallet_docs, mallet_id2word, mallet_corpus, mallet_corpus_csc = load_mallet_corpus(mallet_docs_filepath,
                                                                                       mallet_id2word_filepath,
                                                                                       mallet_corpus_filepath,
                                                                                       mallet_corpus_csc_filepath)

    optimize_interval_candidates = [int(i) for i in optimize_interval_candidates.split(",")]
    topic_alpha_candidates = [float(i) for i in topic_alpha_candidates.split(",")]
    num_topics_candidates = [int(i) for i in num_topics_candidates.split(",")]
    for optimize_interval in optimize_interval_candidates:
        for topic_alpha in topic_alpha_candidates:
            for num_topics in num_topics_candidates:
                model_folder = f"mallet_iterations-{iterations}_" \
                               f"optimize_interval-{optimize_interval}_" \
                               f"topic_alpha-{topic_alpha}_" \
                               f"nun_topics-{num_topics}_lda"
                model_dir = make_dir(os.path.join(models_dir, model_folder))
                mallet_model_filename = f"mallet_{iterations}-{optimize_interval}-{topic_alpha}-{num_topics}_lda"
                mallet_model_filepath = os.path.join(model_dir, mallet_model_filename)
                if not os.path.exists(mallet_model_filepath):
                    train_mallet_lda_model(mallet_docs,
                                           mallet_id2word,
                                           mallet_corpus,
                                           workers,
                                           iterations,
                                           optimize_interval,
                                           topic_alpha,
                                           num_topics,
                                           mallet_model_filepath,
                                           models_coherence_filepath)

                lda_vis_html_filepath = f"{mallet_model_filepath}_vis.html"
                if build_lda_vis and not os.path.exists(lda_vis_html_filepath):
                    save_lda_vis(mallet_model_filepath,
                                 lda_vis_html_filepath,
                                 mallet_corpus_csc,
                                 num_terms_to_display=30,
                                 lambda_step=0.01,
                                 sort_topics=True,
                                 multi_dimensional_scaling="tsne")


def main(lda_config_filepath: str):
    lda_config = read_config_to_dict(lda_config_filepath)
    domain_dir = get_data_filepath(lda_config["domain"])
    topic_modeling_dir = os.path.join(domain_dir, lda_config["topic_modeling_folder"])
    corpus_dir = os.path.join(topic_modeling_dir, lda_config["corpus_folder"])
    models_dir = make_dir(os.path.join(topic_modeling_dir, lda_config["models_folder"]))
    mallet_docs_filepath = os.path.join(corpus_dir, lda_config["mallet_docs_filename"])
    mallet_id2word_filepath = os.path.join(corpus_dir, lda_config["mallet_id2word_filename"])
    mallet_corpus_filepath = os.path.join(corpus_dir, lda_config["mallet_corpus_filename"])
    mallet_corpus_csc_filepath = os.path.join(corpus_dir, lda_config["mallet_corpus_csc_filename"])
    models_coherence_filepath = os.path.join(models_dir, lda_config["models_coherence_filename"])

    train_mallet_lda_models(mallet_docs_filepath,
                            mallet_id2word_filepath,
                            mallet_corpus_filepath,
                            mallet_corpus_csc_filepath,
                            lda_config["workers"],
                            lda_config["iterations"],
                            lda_config["optimize_interval_candidates"],
                            lda_config["topic_alpha_candidates"],
                            lda_config["num_topics_candidates"],
                            lda_config["build_lda_vis"],
                            models_dir,
                            models_coherence_filepath)


if __name__ == "__main__":
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--lda_conf", default="conf/lda_template.cfg", required=False)

    lda_config_filepath = parser.parse_args().lda_conf

    main(lda_config_filepath)
