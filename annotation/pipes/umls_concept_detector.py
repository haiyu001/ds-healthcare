from collections import defaultdict
from typing import Optional, List, Tuple, Dict, Any
from utils.general_util import load_json_file
from utils.resource_util import get_model_filepath
from quickumls import QuickUMLS
from spacy import Language
from spacy.tokens import Doc


class UMLSConceptDetector(object):

    def __init__(self,
                 nlp: Language,
                 quickumls_filepath: Optional[str] = None,
                 overlapping_criteria: str = "score",
                 similarity_name: str = "jaccard",
                 threshold: float = 0.85,
                 window: int = 5,
                 accepted_semtypes: Optional[str] = None,
                 best_match: bool = True,
                 keep_uppercase: bool = False,
                 attrs: Tuple[str] = ("umls_concepts",)):

        self.nlp = nlp
        self.best_match = best_match
        self.keep_uppercase = keep_uppercase
        self.quickumls_filepath = quickumls_filepath or get_model_filepath("UMLS", "QuickUMLS")
        self.accepted_semtypes = self._set_accepted_semtypes(accepted_semtypes)
        self.quickumls = QuickUMLS(quickumls_fp=self.quickumls_filepath,
                                   accepted_semtypes=self.accepted_semtypes,
                                   overlapping_criteria=overlapping_criteria,
                                   similarity_name=similarity_name,
                                   threshold=threshold,
                                   window=window,
                                   spacy_component=True)
        self._umls_concepts, = attrs
        Doc.set_extension(self._umls_concepts, getter=self.get_umls_concepts, force=True)

    def __call__(self, doc: Doc) -> Doc:
        return doc

    def _set_accepted_semtypes(self, accepted_semtypes: Optional[str]) -> List[str]:
        valid_semtypes_dict = load_json_file(get_model_filepath("UMLS", "semtypes.json"))
        valid_semtypes = valid_semtypes_dict.keys()
        if accepted_semtypes:
            accepted_semtypes = accepted_semtypes.split(",")
            for semtype in accepted_semtypes:
                if semtype not in valid_semtypes:
                    raise TypeError(f"{semtype} is not a valid semantic type id.")
        else:
            accepted_semtypes = list(valid_semtypes)
        return accepted_semtypes

    def get_umls_concepts(self, doc: Doc) -> List[Dict[str, Any]]:
        matches = self.quickumls._match(doc, best_match=self.best_match, ignore_syntax=False)
        umls_concepts_dict = defaultdict(list)
        for match in matches:
            for ngram_match_dict in match:
                start_char_idx = int(ngram_match_dict["start"])
                end_char_idx = int(ngram_match_dict["end"])
                cui = ngram_match_dict["cui"]
                concept_span = doc.char_span(start_char_idx, end_char_idx, label=cui)
                concept_key = (concept_span.start, concept_span.end, concept_span.text)
                umls_concepts_dict[concept_key].append({
                    "concept_id": cui,
                    "concept_term": ngram_match_dict["term"],
                    "concept_similarity": round(ngram_match_dict["similarity"], 4),
                    "concept_semtype_ids": ",".join(list(ngram_match_dict["semtypes"])),
                })
        umls_concepts = []
        for concept_key, concept_val in umls_concepts_dict.items():
            umls_concepts.append({
                "start_id": concept_key[0],
                "end_id": concept_key[1],
                "text": concept_key[2],
                "concepts": concept_val,
            })
        return umls_concepts