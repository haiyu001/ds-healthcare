from topic_modeling.bert_topic.bertopic_wrapper import BERTopicWrapper
from utils.general_util import dump_json_file, save_pdf, make_dir
import os


if __name__ == "__main__":

    bert_topic_model_folder, output_folder = "bert_topic_model_3", "test_3"

    bert_topic_model_filepath = f"/Users/haiyang/data/webmd/topic_modeling/{bert_topic_model_folder}/bert_topic_model.pkl"

    bert_topic_model = BERTopicWrapper.load(bert_topic_model_filepath)

    output_dir = make_dir(f"/Users/haiyang/Desktop/bertopic_test/{output_folder}")

    # bert_topic_model.visualize_topics().write_html(os.path.join(output_dir, "bert_topic_vis.html"))

    topic_info_pdf = bert_topic_model.get_topic_info()
    topic_terms_list = []
    for topic_id, term_prob_list in bert_topic_model.get_topics().items():
        topic_terms = []
        for term, prob in term_prob_list:
            topic_terms.append(term)
        topic_terms = " ".join(topic_terms)
        topic_terms_list.append(topic_terms)
    topic_info_pdf["Terms"] = topic_terms_list
    save_pdf(topic_info_pdf, os.path.join(output_dir, "bert_topic_info.csv"))

