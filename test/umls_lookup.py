from utils.general_util import load_json_file
from utils.resource_util import get_model_filepath
import sqlite3


class UMLSLookup(object):
    sqlite = None
    preferred_sources = ['"SNOMEDCT"', '"MTH"']

    def __init__(self):
        self.semantic_types = load_json_file(get_model_filepath("UMLS", "semtypes.json"))
        self.sqlite = sqlite3.connect(get_model_filepath("UMLS", "umls.db"))

    def lookup_code(self, cui, preferred=True):
        """looking cui up in our "descriptions" database. The "preferred" settings has the effect that only names
        from SNOMED (SNOMEDCD) and the Metathesaurus (MTH) will be reported.
        :returns: A list of triples with (STR (Concpet Name), SAB (Abbreviated Source Name), STY(Semantic Type))
        """
        if cui is None or len(cui) < 1:
            return []

        # take care of negations
        negated = cui[0] == "-"
        if negated:
            cui = cui[1:]

        parts = cui.split("@", 1)
        lookup_cui = parts[0]

        if preferred:
            sql = "SELECT STR, SAB, STY FROM descriptions WHERE CUI = ? AND SAB IN ({})".format(
                ", ".join(UMLSLookup.preferred_sources))
        else:
            sql = "SELECT STR, SAB, STY FROM descriptions WHERE CUI = ?"

        arr = []
        for res in self.sqlite.execute(sql, (lookup_cui,)):
            if negated:
                arr.append(("[NEGATED] {}".format(res[0], res[1], res[2])))
            else:
                arr.append(res)
        return arr

    def lookup_code_meaning(self, cui, preferred=True, no_html=True):
        """ Return a string (an empty string if the cui is null or not found) by looking it up in our "descriptions"
        database. The "preferred" settings has the effect that only names from SNOMED (SNOMEDCD) and the
        Metathesaurus (MTH) will be reported.
        """
        names = []
        for res in self.lookup_code(cui, preferred):
            if no_html:
                names.append("{} ({})  [{}]".format(res[0], res[1], res[2]))
            else:
                names.append("{} (<span style=\"color:#090;\">{}</span>: {})".format(res[0], res[1], res[2]))

        comp = ", " if no_html else "<br/>\n"
        return comp.join(names) if len(names) > 0 else ""

    def lookup_code_for_name(self, name, preferred=True):
        """ Tries to find a good concept code for the given concept name. Uses our indexed `descriptions` table.
        :returns: A list of triples with (CUI (Concept-ID), SAB (Abbreviated Source Name), STY (Semantic Type))
        """
        if name is None or len(name) < 1:
            return None

        if preferred:
            sql = "SELECT CUI, SAB, STY FROM descriptions WHERE STR LIKE ? AND SAB IN ({})".format(
                ", ".join(UMLSLookup.preferred_sources))
        else:
            sql = "SELECT CUI, SAB, STY FROM descriptions WHERE STR LIKE ?"
        arr = []
        for res in self.sqlite.execute(sql, ("%" + name + "%",)):
            arr.append(res)
        return arr

    def lookup_sementic_type_name_and_group(self, semantic_type_ids):
        sementic_type_name_and_group_list = []
        semantic_type_ids = semantic_type_ids.split("|")
        for semantic_type_id in semantic_type_ids:
            sementic_type_name_and_group_list.append(f'{lookup.semantic_types[semantic_type_id]["type_name"]} -> '
                                                     f'{lookup.semantic_types[semantic_type_id]["group_name"]}')
        return sementic_type_name_and_group_list


if "__main__" == __name__:

    lookup = UMLSLookup()

    concept_id = "C1737642" # "C0008115" # "C0027051"
    concept_info = lookup.lookup_code(concept_id, preferred=False)
    print(f'\nSearch for "{concept_id}" returns:')
    for concept_name, concept_source, semantic_type_id in concept_info:
        print(f'{concept_name} - {concept_source} - {semantic_type_id} - '
              f'{lookup.lookup_sementic_type_name_and_group(semantic_type_id)}')

    concept_name = "heart attack"
    print(f'\nSearch for "{concept_name}" returns:')
    concept_candidates = lookup.lookup_code_for_name(concept_name, preferred=True)
    for concept_candidate in concept_candidates:
        print("{}:  {}".format(concept_candidate, lookup.lookup_code_meaning(concept_candidate[0])))
