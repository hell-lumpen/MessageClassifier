import numpy
import numpy as np
from pandas.io import pickle
from pandas import DataFrame
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from settings import CONFIG


def get_test_data(dataset: DataFrame):
    x_train, x_test, y_train, y_test, message_train, message_test = train_test_split(dataset['preprocessed'], dataset['tag'], dataset['text'],
                                                        test_size=CONFIG['classifier_settings']['test_size'],
                                                        random_state=CONFIG['classifier_settings']['random_state'],
                                                        stratify=dataset['tag'])
    return x_test.tolist(), y_test.tolist(), message_test.tolist()


class Pair:
    def __init__(self, label: str, proba: numpy.float64):
        self.label = label
        self.proba = proba

    def __str__(self):
        return f'label={self.label}, proba={self.proba.astype(str)}'

    def __repr__(self) -> str:
        return f'\nlabel={self.label}, proba={self.proba.astype(str)}'


class AbstractNeuralNetwork:
    def __init__(self):
        self.model = None
        self.accuracy = None
        self.report = None
        self.tags = None

    def fit(self, dataset: DataFrame, data_source: str):
        self.tags = sorted(dataset['tag'].unique().tolist())

        x_train, x_test, y_train, y_test = train_test_split(dataset[data_source], dataset['tag'],
                                                            test_size=CONFIG['classifier_settings']['test_size'],
                                                            random_state=CONFIG['classifier_settings']['random_state'],
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

    def predict(self, features: list) -> list:
        return self.model.predict(features).tolist()

    def predict_proba(self, features: list):
        probs = self.model.predict_proba(features)
        proba_list = []
        for proba in probs:
            feature_proba = []
            for i, p in enumerate(proba):
                feature_proba.append(Pair(self.tags[i], p))
            feature_proba.sort(key=lambda x: x.proba, reverse=True)
            proba_list.append(feature_proba)

        return proba_list

    def get_accuracy(self) -> str:
        return self.accuracy

    def get_report(self) -> str:
        return self.report

    def get_model_name(self) -> str:
        return 'AbstractNeuralNetwork'

    def get_my_report(self, test_features: list, test_labels: list):
        predicted_proba = self.model.predict_proba(test_features)
        wrong_answers = 0
        wrong_answers_2 = 0

        for i, test_label in enumerate(test_labels):
            if test_label != predicted_proba[i][0]:
                wrong_answers += 1
                if test_label != predicted_proba[i][1]:
                    wrong_answers_2 += 1

        # print(f'model name : {self.model.get_model_name()}')
        print(f'Answer accuracy {1 - (wrong_answers / len(test_features))}')
        print(f'Answer accuracy (among the first two most likely answers) {1 - (wrong_answers_2 / len(test_features))}')
