from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

from .abstract_neural_network import AbstractNeuralNetwork


class LinSupport(AbstractNeuralNetwork):
    def __init__(self):
        super().__init__()
        self.model = Pipeline([('vect', CountVectorizer()),
                               ('tfidf', TfidfTransformer()),
                               ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3,
                                                     random_state=42,max_iter=5, tol=None))])
