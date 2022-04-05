from annotation.annotation_utils.annotate_util import doc_to_dict, DEFAULT_SPACY_PACKAGE
from annotation.annotation_utils.pipeline_util import get_nlp_model
from pprint import pprint
import json

dummy_normalizer_config = {
    "merge_words": {"battery life": {"merge": "batterylife", "type": "canonical"}},
    "split_words": {"autonomouscars": "autonomous cars"},
    "replace_words": {"thisr": "these"},
}

nlp_model_config = dict(
    use_gpu=False,
    lang="en",
    spacy_package=DEFAULT_SPACY_PACKAGE,
    text_meta_config={"text_fields_in_json": ["content"], "meta_fields_to_keep": ["record_id"]},
    preprocessor_config={},
    stanza_base_tokenizer_package="default",
    normalizer_config=dummy_normalizer_config,
    stanza_pipeline_config={"processors": "tokenize,ner,sentiment"},
    spacy_pipeline_config={"exclude": ["ner"]},
    custom_pipes_config=[
        ("phrase_detector", {}),
        ("lang_detector", {}),
    ],
)

if __name__ == "__main__":

    nlp = get_nlp_model(**nlp_model_config)

    content = "Thisr  autonomouscars have good battery life in China. I hate that laptop."

    content_meta = json.dumps({"record_id": "001", "source": "dummy", "content": content})

    docs = nlp.pipe([content, content_meta], n_process=2)
    for doc in docs:
        pprint(doc_to_dict(doc))
        print("-" * 100)
