from annotation.tokenization.preprocessor import Preprocessor
from annotation.annotation_utils.annotation_util import DEFAULT_SPACY_PACKAGE
from annotation.components.annotator import get_nlp_model, doc_to_dict
import json
import pandas as pd
import collections
import os


preprocessor = Preprocessor()
test_dir = "/Users/haiyang/Desktop/annotation"

filepath = os.path.join(test_dir, "test_review.csv")
file_df = pd.read_csv(filepath, encoding="utf-8")
file_df["content"] = file_df["content"].apply(lambda x: preprocessor.preprocess(x))
file_df["words_cnt"] = file_df["content"].apply(lambda x: len(x.split()))
file_df = file_df[(file_df["words_cnt"] > 15) & (file_df["words_cnt"] < 150)]
file_df.to_json(os.path.join(test_dir, "tmp.json"), orient="records", lines=True, force_ascii=False)
print(file_df.shape)


nlp_model_config = dict(
    lang="en",
    spacy_package=DEFAULT_SPACY_PACKAGE,
    text_meta_config={"text_fields_in_json": ["content"]},
    preprocessor_config={},
    stanza_base_tokenizer_package=None,
    normalizer_config=None,
    spacy_pipeline_config=None,
    stanza_pipeline_config={},
    custom_pipes_config=None
)

nlp = get_nlp_model(**nlp_model_config)
input_filepath = os.path.join(test_dir, "tmp.json")
output_filepath = os.path.join(test_dir, "stanza_annotation.json")
with open(output_filepath, "w") as output:
    with open(input_filepath) as input:
        cnt = 0
        docs = nlp.pipe(input)
        for doc in docs:
            output.write(json.dumps(doc_to_dict(doc)) + "\n")
            cnt += 1
            if cnt % 1000 == 0:
                print(cnt)

pos_stanza_to_spacy = collections.defaultdict(collections.Counter)
tag_stanza_to_spacy = collections.defaultdict(collections.Counter)
rel_stanza_to_spacy = collections.defaultdict(collections.Counter)
rel_spacy_to_stanza = collections.defaultdict(collections.Counter)

spacy_annotation_filepath = os.path.join(test_dir, "spacy_annotation.json")
stanza_annotation_filepath = os.path.join(test_dir, "stanza_annotation.json")
with open(spacy_annotation_filepath) as spacy_annotation:
    with open(stanza_annotation_filepath) as stanza_annotation:
        count = 0
        spacy_lines, stanza_lines = spacy_annotation.readlines(), stanza_annotation.readlines()
        for spacy_line, stanza_line in zip(spacy_lines, stanza_lines):
            spacy_record, stanza_record = json.loads(spacy_line), json.loads(stanza_line)
            if spacy_record["_"]["metadata"]["record_id"] != stanza_record["_"]["metadata"]["record_id"]:
                break
            for spacy_token, stanza_token in zip(spacy_record["tokens"], stanza_record["tokens"]):
                spacy_pos, stanza_pos = spacy_token["pos"], stanza_token["pos"]
                spacy_tag, stanza_tag = spacy_token["tag"], stanza_token["tag"]
                spacy_rel, stanza_rel = spacy_token["rel"], stanza_token["rel"]
                pos_stanza_to_spacy[stanza_pos][spacy_pos] += 1
                tag_stanza_to_spacy[stanza_tag][spacy_tag] += 1
                rel_stanza_to_spacy[stanza_rel][spacy_rel] += 1
                rel_spacy_to_stanza[spacy_rel][stanza_rel] += 1
            count += 1
            if count % 1000 == 0:
                print(count)


def save_mapping(mapping_dict, mapping_type, most_common, save_dir):
    mapping_dict = {k: dict(v.most_common(most_common)) for k, v in mapping_dict.items()}
    mapping_dict = collections.OrderedDict(sorted(mapping_dict.items()))
    with open(os.path.join(save_dir, f"{mapping_type}.json"), "w") as output:
        json.dump(mapping_dict, output, indent=2)

save_mapping(pos_stanza_to_spacy, "pos_stanza_to_spacy", 3, test_dir)
save_mapping(tag_stanza_to_spacy, "tag_stanza_to_spacy", 3, test_dir)
save_mapping(rel_stanza_to_spacy, "rel_stanza_to_spacy", 10, test_dir)
save_mapping(rel_spacy_to_stanza, "rel_spacy_to_stanza", 10, test_dir)