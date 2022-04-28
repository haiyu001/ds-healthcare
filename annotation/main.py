from annotation.annotation_utils.annotator_util import read_nlp_model_config, get_canonicalization_nlp_model_config
from annotation.components.annotator import doc_to_dict, get_nlp_model
from utils.resource_util import get_repo_dir
from utils.general_util import setup_logger
from pprint import pprint
import json
import os

if __name__ == "__main__":
    setup_logger()

    dummy_normalizer_config = {
        "replace_norm": {
            "thesr": {
                "key": "thesr",
                "value": "these",
                "case_insensitive": True,
            }
        },
        "merge_norm": {
            "battery life": {
                "key": "battery life",
                "value": "batterylife",
                "case_insensitive": True,
            }
        },
        "split_norm": {
            "autonomouscars": {
                "key": "autonomouscars",
                "value": "autonomous cars",
                "case_insensitive": True,
            }
        },
    }

    nlp_model_config_filepath = os.path.join(get_repo_dir(), "annotation", "pipelines", "conf/nlp_model_template.cfg")

    # nlp_model_config = get_canonicalization_nlp_model_config(nlp_model_config_filepath)

    nlp_model_config = read_nlp_model_config(nlp_model_config_filepath)
    nlp_model_config["normalizer_config"].update(dummy_normalizer_config)

    nlp = get_nlp_model(**nlp_model_config)

    sample_1 = "Thesr  autonomouscars have good battery life in China. I hatte that laptop in 5yrs and cool."
    # sample_1 = "I struggle with depression, anxiety and panic attacks in 3 month."
    # sample_1 = "I'm not done my 3 month sample but I will be switching after the 3 months are done."
    sample_2 = json.dumps({"record_id": "001", "source": "dummy", "content": "He had a huge heart attack."})

    docs = nlp.pipe([sample_1, sample_2], n_process=1)
    for doc in docs:
        pprint(doc_to_dict(doc))
        print("-" * 100)
