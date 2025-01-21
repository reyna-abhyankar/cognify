from openai import OpenAI
from abc import ABC, abstractmethod
from typing import Any
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pickle

import logging

logger = logging.getLogger(__name__)

class Cluster(ABC):
    
    def __init__(self, members: list[int]):
        """
        members: list of indices of samples in the dataset
        """
        self.members = members
    
    @abstractmethod
    def dist(self, other: 'Cluster') -> float:
        """Compute the distance between two clusters
        """
        pass


class DomainManagerInterface(ABC):
    _clusters: list[Cluster]
    
    def __init__(self):
        self._clusters = []
    
    @abstractmethod
    def partition(self, samples: list[Any]):
        """Partition a given dataset into clusters
        """
        pass

    @abstractmethod
    def belongs_to(self, sample: Any) -> int:
        """Return the cluster index that the sample belongs to
        """
        pass
    
    def save(self, filepath: str) -> None:
        """Stores the cluster object to a file with class metadata."""
        data = {
            'class': self.__class__.__name__,
            'object': self,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Input partition stored to {filepath}")

    @staticmethod
    def load(filepath: str):
        """Loads a cluster object from a file using class metadata."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        class_name = data['class']
        cluster_object = data['object']

        # Dynamic class resolution
        if class_name in globals():
            cls = globals()[class_name]
            if not isinstance(cluster_object, cls):
                raise TypeError(f"Object in file is not of type {class_name}")
            logger.info(f"{class_name} loaded from {filepath}")
            return cluster_object
        else:
            raise ValueError(f"Unknown class name: {class_name}")

class KMeansCluster(Cluster):
    def __init__(self, members: list[int], centroid: np.ndarray):
        super().__init__(members)
        self.centroid = centroid
    
    def dist(self, other: Cluster) -> float:
        return sum((self.centroid - other.centroid) ** 2) ** 0.5
    
    
class TextEmbeddingDomainManager(DomainManagerInterface):
    _clusters: list[KMeansCluster]
    
    def __init__(self, n_groups, seed=2024, parallel=50, embedding_func=None):
        super().__init__()
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
            
        cluster_ids = engine.predict(embeddings)
        for i in range(self.n_groups):
            self._clusters.append(
                KMeansCluster(
                    [j for j in range(len(samples)) if cluster_ids[j] == i], 
                    engine.cluster_centers_[i]
                )
            )
    
    def _belongs_to_by_embedding(self, sample: str) -> int:
        embedding = self.embedding_func(sample)
        return self._engine.predict([embedding])[0]

    def partition(self, samples: list) -> list[int]:
        self._partition_by_embedding(samples)
    
    def belongs_to(self, sample: str) -> int:
        return self._belongs_to_by_embedding(sample)