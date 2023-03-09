from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline

from .abstract_neural_network import AbstractNeuralNetwork


class MyNearestNeighbors(AbstractNeuralNetwork):
    def __init__(self) -> None:
        super().__init__()
        self.model = Pipeline([('vect', CountVectorizer()),
                               ('tfidf', TfidfTransformer()),
                               ('clf', NearestCentroid())])
