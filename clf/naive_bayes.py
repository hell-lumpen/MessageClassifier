from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .abstract_neural_network import AbstractNeuralNetwork


class NaiveBayes(AbstractNeuralNetwork):
    def __init__(self) -> None:
        super().__init__()
        self.model = Pipeline([('vect', CountVectorizer()),
                               ('tfidf', TfidfTransformer()),
                               ('clf', MultinomialNB())])

    def get_model_name(self) -> str:
        return 'MultinomialNB'
