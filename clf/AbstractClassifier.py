class AbstractClassifier:
    def fit(self, features: list, labels: list):
        pass

    def predict(self, text_features: list):
        pass

    def predict_proba(self, text_features: list) -> list:
        pass
