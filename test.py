import pandas

from clf import LogRegression, LinSupport, NaiveBayes, RandomForest, CalibratedClassifier
from settings import CONFIG
from MessagePreprocessor import MessagePreprocessor

# dataset = pandas.DataFrame(pandas.read_csv(CONFIG['preprocessor_settings']['raw_data_file'],
#                                            usecols=['text', 'tag'], encoding='utf8'))
#
# preprocessor = MessagePreprocessor()
#
# dataset = preprocessor.preprocess_dataset(dataset, destination_label='raw', lemmatize=False, remove_stopwords=False)
# dataset = preprocessor.preprocess_dataset(dataset, destination_label='!stopwords', lemmatize=False)
# dataset = preprocessor.preprocess_dataset(dataset, destination_label='stemmed', lemmatize=False, stemming=True)
# dataset = preprocessor.preprocess_dataset(dataset, destination_label='preprocessed', lemmatize=True)
#
# dataset = dataset[dataset['preprocessed'].notna()]
# dataset.to_csv(CONFIG['preprocessor_settings']['prep_data_file'], header=True)

dataset = pandas.DataFrame(pandas.read_csv(CONFIG['preprocessor_settings']['prep_data_file'], encoding='utf8'))


models = [NaiveBayes(), LogRegression(), LinSupport(), RandomForest(), CalibratedClassifier()]
columns = ['raw', '!stopwords', 'stemmed', 'preprocessed']

for model in models:
    print(model.get_model_name())
    for col in columns:
        model.fit(dataset, data_source=col)
        print(col + ' ' + model.get_accuracy().__str__())
    print()


# print(dataset.iloc[1]['text'])
# print(dataset['tag'].value_counts())
# tags = dataset['tag'].unique().tolist()
# for tag in tags:
    # preprocessed_dataset = preprocessed_dataset.append(dataset[dataset['tag'] == tag][:100], ignore_index=True)
