from pymetamap import MetaMapLite, MetaMap
from utils.resource_util import get_model_filepath

if __name__ == "__main__":

    # ************************************************************************************************************
    # * Recognizing UMLS Concepts in Text by MetaMap or MataMapLite
    # * MetaMap settings:
    #   run "~/ds_models/UMLS/MetaMap/public_mm/bin/skrmedpostctl start" to start SKR/Medpost POS Tagger Server
    #   run "~/ds_models/UMLS/MetaMap/public_mm/bin/skrmedpostctl stop" to stop SKR/Medpost POS Tagger Server
    # * MataMapLite settings:
    #   edit "~/ds_models/UMLS/MetaMap/public_mm_lite/config/metamaplite.properties" to set correct model path
    # ************************************************************************************************************

    samples = ["Heart Attack", "He had a huge heart attack"]

    # mm = MetaMap.get_instance(get_model_filepath("UMLS", "MetaMap", "public_mm", "bin", "metamap18"))
    # concepts, error = mm.extract_concepts(sents, [1, 2])
    # for concept in concepts:
    #     print(concept)

    mm_lite = MetaMapLite.get_instance(get_model_filepath("UMLS", "MetaMap", "public_mm_lite"))
    concepts, error = mm_lite.extract_concepts(samples, [1, 2])
    for concept in concepts:
        print(concept)
