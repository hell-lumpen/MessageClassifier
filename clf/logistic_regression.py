from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .abstract_neural_network import AbstractNeuralNetwork


class LogRegression(AbstractNeuralNetwork):
    def __init__(self) -> None:
        super().__init__()
        self.model = Pipeline([('vect', CountVectorizer()),
                               ('tfidf', TfidfTransformer()),
                               ('clf', LogisticRegression(n_jobs=1,
                                                          C=1e5,
                                                          max_iter=100,
                                                          solver='saga',
                                                          multi_class='multinomial'
                                                          ))])
