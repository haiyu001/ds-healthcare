from annotation.annotation_utils.annotation_util import read_nlp_model_config
from annotation.components.annotator import Annotator, doc_to_dict
from utils.resource_util import get_repo_dir
from pprint import pprint
import json
import os

if __name__ == "__main__":

    dummy_normalizer_config = {
        "merge_words": {"battery life": {"merge": "batterylife", "type": "canonical"}},
        "split_words": {"autonomouscars": "autonomous cars"},
        "replace_words": {"thesr": "these"},
    }

    nlp_model_config_filepath = os.path.join(get_repo_dir(), "conf", "nlp_model_template.cfg")
    nlp_model_config = read_nlp_model_config(nlp_model_config_filepath)
    nlp_model_config["normalizer_config"] = dummy_normalizer_config

    nlp = Annotator(**nlp_model_config).nlp

    sample_1 = "Thesr  autonomouscars have good battery life in China. I hatte that laptop."
    sample_1 = "I struggle with depression, anxiety and panic attacks and more depression in 3 month."
    sample_2 = json.dumps({"record_id": "001", "source": "dummy", "content": "He had a huge heart attack."})

    docs = nlp.pipe([sample_1, sample_2], n_process=1)
    for doc in docs:
        pprint(doc_to_dict(doc))
        print("-" * 100)
