# from tmp.umls_metamap.MetaMapLite import MetaMapLite
from pymetamap import MetaMapLite
from pymetamap import MetaMap

sents = ["Heart Attack", "John had a huge heart attack"]

# run "/Users/haiyang/ds_models/UMLS/MetaMap/public_mm/bin/skrmedpostctl start" to start SKR/Medpost POS Tagger Server
# run "/Users/haiyang/ds_models/UMLS/MetaMap/public_mm/bin/skrmedpostctl stop" to stop SKR/Medpost POS Tagger Server
mm = MetaMap.get_instance("/Users/haiyang/ds_models/UMLS/MetaMap/public_mm/bin/metamap18")
concepts, error = mm.extract_concepts(sents, [1, 2])
for concept in concepts:
    print(concept)

# mm_lite = MetaMapLite.get_instance("/Users/haiyang/ds_models/UMLS/MetaMap/public_mm_lite/")
# concepts, error = mm_lite.extract_concepts(sents, [1, 2])
# for concept in concepts:
#     print(concept)
