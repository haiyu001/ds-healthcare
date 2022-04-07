from annotation.annotation_utils.annotation_util import read_annotation_config
from annotation.components.annotator import Annotator, doc_to_dict
from pathlib import Path
from pprint import pprint
import json
import os

if __name__ == "__main__":

    dummy_normalizer_config = {
        "merge_words": {"battery life": {"merge": "batterylife", "type": "canonical"}},
        "split_words": {"autonomouscars": "autonomous cars"},
        "replace_words": {"thisr": "these"},
    }

    annotation_config_filepath = os.path.join(Path(__file__).parent, "annotation.cfg")
    nlp_model_config = read_annotation_config(annotation_config_filepath)
    nlp_model_config["normalizer_config"].update(dummy_normalizer_config)

    nlp = Annotator(**nlp_model_config).nlp

    content = "Thisr  autonomouscars have good battery life in China. I hate that laptop."

    content_meta = json.dumps({"record_id": "001", "source": "dummy", "content": content})

    docs = nlp.pipe([content, content_meta], n_process=2)
    for doc in docs:
        pprint(doc_to_dict(doc))
        print("-" * 100)
