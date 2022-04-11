# from tmp.umls_metamap.MetaMapLite import MetaMapLite
from pymetamap import MetaMapLite

sents = ["Heart Attack", "John had a huge heart attack"]

# run ./bin/skrmedpostctl start for MetaMap
# mm = MetaMap.get_instance("/Users/haiyang/ds_models/UMLS/MetaMap/public_mm/bin/metamap18")
# concepts, error = mm.extract_concepts(sents, [1, 2])
# for concept in concepts:
#     print(concept)

mm_lite = MetaMapLite.get_instance("/Users/haiyang/ds_models/UMLS/MetaMap/public_mm_lite/")
concepts, error = mm_lite.extract_concepts(sents, [1, 2])
for concept in concepts:
    print(concept)
