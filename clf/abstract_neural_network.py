import numpy as np
from pandas.io import pickle
from pandas import DataFrame
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from settings import CONFIG


class AbstractNeuralNetwork:
    def __init__(self):
        self.model = None
        self.accuracy = None
        self.report = None
        self.tags = None

    def fit(self, dataset: DataFrame):
        self.tags = sorted(dataset['tag'].unique().tolist())
        x_train, x_test, y_train, y_test = train_test_split(dataset['preprocessed'], dataset['tag'],
                                                            test_size=CONFIG['classifier_settings']['test_size'],
                                                            random_state=42,
                                                            stratify=dataset['tag'])
        self.model.fit(x_train, y_train)

        y_predicted = self.model.predict(x_test)
        self.accuracy = accuracy_score(y_predicted, y_test)
        self.report = classification_report(y_test, y_predicted, target_names=dataset['tag'].unique())

    def save(self, path: str) -> bool:
        if self.model is None:
            return False

        with open(path, 'wb') as file:
            pickle.dump(self.model, file)
            return True

    def load(self, path: str):
        with open(path, 'rb') as file:
            self.model = pickle.load(file)

    def predict(self, text):
        return self.model.predict(text)

    def predict_proba(self, text):
        probs = self.model.predict_proba(text)
        arr = []
        for i in range(len(probs)):
            arr.append(np.column_stack((probs[i], self.tags)))
        return arr

    def get_accuracy(self) -> str:
        return self.accuracy

    def get_report(self) -> str:
        return self.report
