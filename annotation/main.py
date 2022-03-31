from annotation.annotation_utils.annotation import doc_to_dict
from annotation.annotation_utils.pipeline import get_nlp_model
from pprint import pprint
import json

nlp_model_config = dict(
    lang="en",
    spacy_package="en_core_web_md-3.2.0",
    text_meta_config={"text_fields_in_json": ["content"]},
    preprocessor_config={},
    stanza_base_tokenizer_package=None,
    normalizer_config={"merge_words": {"battery life": {"merge": "batterylife", "type": "canonical"}},
                       "split_words": {"autonomouscars": "autonomous cars"},
                       "replace_words": {"thisr": "these"}},
    spacy_pipeline_config={},
    stanza_pipeline_config=None,
    custom_pipes_config=[("language_detector", {}),
                         ("phrase_detector", {})]
)

nlp = get_nlp_model(**nlp_model_config)

content = "Thisr  autonomouscars have good battery life in the U.S."

content_meta = json.dumps({"record_id": "001", "source": "dummy", "content": content})

docs = nlp.pipe([content, content_meta])
for doc in docs:
    print('-' * 100)
    pprint(doc_to_dict(doc))
