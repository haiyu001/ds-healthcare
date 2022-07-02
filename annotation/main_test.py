from annotation.annotation_utils.annotator_util import read_nlp_model_config, get_canonicalization_nlp_model_config
from annotation.components.annotator import doc_to_dict, get_nlp_model, get_nlp_model_config_str
from utils.general_util import setup_logger, get_repo_dir
from pprint import pprint
import logging
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
                "value": "battery_life",
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
    logging.info(f"nlp model config (use_gpu = {nlp_model_config['use_gpu']}):\n{get_nlp_model_config_str(nlp)}")

    samples = [
        # "Thesr  autonomouscars have kind-of good battery life if not in China. uh-uh I sort-of hatje that laptop in 5yrs and cool.",
        # "I don't struggle with depression, anxiety and panic attacks in 3 month.",
        json.dumps({"record_id": "001", "source": "dummy", "Reviews": "He had no sign of huge heart attack."}),
    ]

    docs = nlp.pipe(samples, n_process=1)
    for doc in docs:
        pprint(doc_to_dict(doc))
        print("-" * 100)
