from openai import OpenAI
from abc import ABC, abstractmethod
from typing import Any
from concurrent.futures import ThreadPoolExecutor

class DomainManagerInterface(ABC):
    
    def partition(self, samples: list[Any]) -> list[int]:
        """Partition a given dataset into clusters
        
        Returns:
            list[int]: The cluster indices for each sample
        """
        pass
    
    
class TextEmbeddingDomainManager(DomainManagerInterface):
    def __init__(self, n_groups, embedding_func=None, seed=2024, parallel=50):
        self.n_groups = n_groups
        self.embedding_func = embedding_func or self.default_get_embedding
        self.seed = seed
        self.parallel = parallel
        self._engine = None

    @staticmethod
    def default_get_embedding(text: str):
        client = OpenAI()
        text = text.replace("\n", " ")
        return client.embeddings.create(input = [text], model='text-embedding-3-small').data[0].embedding

    def _partition_by_embedding(
        self,
        samples: list[str], 
    ):
        """
        Encode the input data into the embedding space then do k-means clustering
        """
        # get embeddings
        with ThreadPoolExecutor(max_workers=self.parallel) as executor:
            embeddings = executor.map(self.embedding_func, samples)
            embeddings = list(embeddings)
            
        # group embeddings
        if self._engine is None:
            from sklearn.cluster import KMeans
            engine = KMeans(n_clusters=self.n_groups, random_state=self.seed)
            engine.fit(embeddings)
            self._engine = engine
        return engine.predict(embeddings)

    def partition(self, samples: list) -> list[int]:
        return self._partition_by_embedding(samples)