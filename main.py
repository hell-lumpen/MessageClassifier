import pandas

from sklearn.model_selection import train_test_split
from clf.CBClassifier import CBClassifier
from settings import CONFIG


def main():
    # msg_processor = MessagePreprocessor(CONFIG['classifier_settings']['row_data_file'])
    # # msg_processor.preprocess()
    dataset = pandas.DataFrame(pandas.read_csv(CONFIG['preprocessor_settings']['prep_data_file'],
                                               usecols=['preprocessed', 'tag'], encoding='utf8'))

    features = dataset['preprocessed']
    labels = dataset['tag']
    model = CBClassifier()
    model.fit(features, labels)
    print(model.predict(features))



if __name__ == "__main__":
    from catboost import Pool, CatBoostClassifier

    dataset = pandas.DataFrame(pandas.read_csv(CONFIG['preprocessor_settings']['prep_data_file'],
                                               usecols=['preprocessed', 'tag'], encoding='utf8'))

    x_train = dataset['preprocessed'].tolist()[:900]
    x_train = [[x] for x in x_train]

    y_train = dataset['tag'].tolist()[:900]
    y_train = [[y] for y in y_train]

    x_test = dataset['preprocessed'].tolist()[900:1030]
    x_test = [[x] for x in x_test]

    y_test = dataset['tag'].tolist()[900:1030]
    y_test = [[y] for y in y_test]

    cat_features = [0]

    train_dataset = Pool(data=x_train,
                         label=y_train,
                         cat_features=cat_features)

    eval_dataset = Pool(data=x_test,
                        label=y_test,
                        cat_features=cat_features)

    model = CatBoostClassifier(iterations=47,
                               learning_rate=0.3)
    # Fit model
    model.fit(X=x_train, y=y_train, cat_features=cat_features, eval_set=(x_test, y_test), logging_level='Silent')
    # Get predicted classes
    preds_class = model.predict(['нужно прописываться общежитие выписываться основный место жительство'])
    # Get predicted probabilities for each class
    preds_proba = model.predict_proba(['нужно прописываться общежитие выписываться основный место жительство'])

    print(preds_class)
    print(preds_proba)
