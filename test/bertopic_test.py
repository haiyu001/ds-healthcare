import hdbscan
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
import os
from topic_modeling.bert_topic.bertopic_wrapper import BERTopicWrapper
from utils.general_util import make_dir, save_pdf

docs = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))["data"]

ngram_range = (1, 1)
min_topic_size = 10
low_memory = False

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
print(type(sentence_model))
embeddings = sentence_model.encode(docs, show_progress_bar=False)

umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", low_memory=low_memory)
hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=min_topic_size, metric="euclidean",
                                cluster_selection_method="eom", prediction_data=True)
vectorizer_model = CountVectorizer(ngram_range=ngram_range, stop_words="english")

topic_model = BERTopic(language="english",
                       top_n_words=10,
                       n_gram_range=ngram_range,
                       min_topic_size=min_topic_size,
                       nr_topics=None,
                       low_memory=low_memory,
                       calculate_probabilities=False,
                       diversity=None,
                       seed_topic_list=None,
                       embedding_model=sentence_model,
                       umap_model=umap_model,
                       hdbscan_model=hdbscan_model,
                       vectorizer_model=vectorizer_model,
                       verbose=False)

topics, probs = topic_model.fit_transform(docs, embeddings)

topic_model.save("/Users/haiyang/Desktop/bertopic/topic_model")

print(topic_model.get_topic_info())

print(topic_model.get_topic(0))

fig = topic_model.visualize_topics()

fig.write_html("/Users/haiyang/Desktop/bertopic/topic_vis.html")








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
