import pandas

from clf import *
from settings import CONFIG


def main():
    # msg_processor = MessagePreprocessor(CONFIG['preprocessor_settings']['row_data_file'])
    # msg_processor.preprocess()
    dataset = pandas.DataFrame(pandas.read_csv(CONFIG['preprocessor_settings']['prep_data_file'],
                                               usecols=['text', 'preprocessed', 'tag'],
                                               encoding='utf8'))


    models = [RandomForest(), LogRegression(), CalibratedClassifier()]
    for model in models:
        model.fit(dataset, 'preprocessed')

    test_features, test_labels, test_message = get_test_data(dataset)
    predicted_labels = []
    for model in models:
        predicted_labels.append(model.predict(test_features))
        predicted_proba = model.predict_proba(test_features)
        wrong_answers = 0
        wrong_answers_2 = 0

        for i, test_label in enumerate(test_labels):
            if test_label != predicted_proba[i][0].label:
                wrong_answers += 1
                if test_label != predicted_proba[i][1].label:
                    wrong_answers_2 += 1

        print(f'model name : {model.get_model_name()}')
        print(f'Answer accuracy {1 - (wrong_answers / len(test_features))}')
        print(f'Answer accuracy (among the first !two! most likely answers) {1 - (wrong_answers_2 / len(test_features))}')

    total_errors = 0
    models_test = pandas.DataFrame(list(zip(test_message, test_features, test_labels, predicted_labels[0], predicted_labels[1], predicted_labels[2])),
                                   columns=['message', 'test_features', 'test_labels', 'RandomForest', 'LogRegression', 'CalibratedClassifier'])

    models_test.to_csv('data/tests.csv', header=True)
    category_error_stat = {}
    for i in range(len(test_labels)):
        if predicted_labels[0][i] != test_labels[i] \
                and predicted_labels[1][i] != test_labels[i] \
                and predicted_labels[2][i] != test_labels[i]:

            try:
                category_error_stat[test_labels[i]] += 1
            except KeyError:
                category_error_stat[test_labels[i]] = 1
            total_errors += 1
    print(f"total errors = {total_errors.__str__()} / {test_labels.__len__()}\n"
          f"max_accuracy = {1 - (total_errors/test_labels.__len__())}")

    print('\nErrors:')
    error_stat = dict(sorted(category_error_stat.items(), reverse=True,
                             key=lambda item: item[1]))

    for key, value in error_stat.items():

        # x.append(key)
        # y.append((value / total_errors) - (dataset["tag"].tolist().count(key) / len(dataset["tag"])))
        print(f'label= {key:30} error_count={value:3} | % in errors={value / total_errors:>.6f} | % in all messages={value / len(test_features):>.6f}')
    #  % in test={dataset["tag"].tolist().count(key) / len(dataset["tag"]):>.6f} |
    # fig, ax = plt.subplots(figsize=(20, 16))
    # ax.plot(x, y, marker='o')
    #
    # ax.set_xlabel('categories')
    # ax.set_ylabel('probability difference in the test and training samples')
    #
    # ax.axhline(y=0, color='r', linestyle='-')
    # ax.tick_params(axis='x', rotation=70)
    #
    # plt.savefig('data/probability_difference.png')
    # plt.show()

if __name__ == "__main__":
    main()