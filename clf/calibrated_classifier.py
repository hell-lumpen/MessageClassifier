from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .abstract_neural_network import AbstractNeuralNetwork


class CalibratedClassifier(AbstractNeuralNetwork):
    def __init__(self):
        super().__init__()
        self.model = Pipeline([('vect', CountVectorizer()),
                               ('tfidf', TfidfTransformer()),
                               ('clf',
                                   CalibratedClassifierCV(
                                       LinearSVC(
                                           C=3,
                                           intercept_scaling=1,
                                           class_weight='balanced',
                                           random_state=42
                                       ),
                                       method='sigmoid',
                                       cv=5
                                   ))
                               ])

    def get_model_name(self) -> str:
        return 'CalibratedClassifierCV'
