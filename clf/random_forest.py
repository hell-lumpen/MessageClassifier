from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

from .abstract_neural_network import AbstractNeuralNetwork


class RandomForest(AbstractNeuralNetwork):
    def __init__(self) -> None:
        super().__init__()
        self.model = Pipeline([('vect', CountVectorizer()),
                               ('tfidf', TfidfTransformer()),
                               ('clf', ExtraTreesClassifier(max_depth=60, random_state=0))])
