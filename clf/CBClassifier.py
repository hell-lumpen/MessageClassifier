from clf.AbstractClassifier import AbstractClassifier
from catboost import CatBoostClassifier, Pool

class CBClassifier(AbstractClassifier):
    def __init__(self):
        self.model = CatBoostClassifier(iterations = 50, learning_rate=0.1, loss_function='MultiClass')
    def fit(self, features: list, labels: list):
        # training_pool = Pool(data=features, label=labels, cat_features=cat_features)
        # self.model.fit(training_pool, cat_features=cat_features)
        self.model.fit(X=features, y=labels)

    def predict(self, text_features: list):
        self.model.predict(text_features)