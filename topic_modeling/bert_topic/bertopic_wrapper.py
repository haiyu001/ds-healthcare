from typing import List, Optional, Union, Tuple
import pandas as pd
from bertopic import BERTopic
import numpy as np
from bertopic.backend._utils import select_backend


class BERTopicWrapper(BERTopic):

    def fit_transform(self,
                      documents: pd.DataFrame,
                      reduced_embeddings: Optional[np.ndarray] = None,
                      y: Optional[Union[List[int], np.ndarray]] = None) -> Tuple[List[int], Optional[np.ndarray]]:

        if self.embedding_model is not None:
            self.embedding_model = select_backend(self.embedding_model, language=self.language)

        # Cluster reduced embeddings
        documents, probabilities = self._cluster_embeddings(reduced_embeddings, documents)

        # Sort and Map Topic IDs by their frequency
        if not self.nr_topics:
            documents = self._sort_mappings_by_frequency(documents)

        # Extract topics by calculating c-TF-IDF
        documents = documents.rename(columns={"Document": "Representation", "Representation": "Document"})
        self._extract_topics(documents)

        # Reduce topics
        if self.nr_topics:
            documents = self._reduce_topics(documents)
        documents = documents.rename(columns={"Document": "Representation", "Representation": "Document"})

        self._map_representative_docs(original_topics=True)
        probabilities = self._map_probabilities(probabilities, original_topics=True)
        predictions = documents.Topic.to_list()

        return predictions, probabilities

    def _preprocess_text(self, documents: np.ndarray) -> List[str]:
        return [document for document in documents]
