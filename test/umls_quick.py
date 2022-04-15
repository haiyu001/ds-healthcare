from annotation.annotation_utils.annotation_util import DEFAULT_SPACY_PACKAGE
from utils.resource_util import get_model_filepath, get_spacy_model_path
from quickumls import QuickUMLS

from pprint import pprint
import spacy

quickumls_dir = get_model_filepath("UMLS", "QuickUMLS")

matcher = QuickUMLS(quickumls_dir, threshold=0.9, spacy_component=True)

nlp = spacy.load(get_spacy_model_path("en", DEFAULT_SPACY_PACKAGE))

text = "He had a huge Heart attack"

# doc = nlp(text)
# matches = matcher._match(doc, best_match=True)
# for match in matches:
#     for ngram_match_dict in match:
#         pprint(ngram_match_dict)
#         print(type(ngram_match_dict["semtypes"]))

nlp.add_pipe()
