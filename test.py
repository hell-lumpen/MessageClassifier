import pandas
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

from clf import LogRegression, LinSupport, NaiveBayes, RandomForest
from settings import CONFIG
from MessagePreprocessor import MessagePreprocessor

dataset = pandas.DataFrame(pandas.read_csv(CONFIG['preprocessor_settings']['prep_data_file'],
                                           usecols=['text', 'preprocessed', 'tag'], encoding='utf8'))

res_dataset = pandas.DataFrame()
tags = dataset['tag'].unique().tolist()
for tag in tags:
    res_dataset = res_dataset.append(dataset[dataset['tag'] == tag][:50], ignore_index=True)

preprocessor = MessagePreprocessor()
# res_dataset = preprocessor.preprocess_dataset(res_dataset, source_label='stemmed', lemmatize=False)

model = LogRegression()
model.fit(preprocessor.preprocess_dataset(res_dataset, source_label='stemmed', lemmatize=False),
          data_source='stemmed')

model2 = LogRegression()
model2.fit(res_dataset, data_source='preprocessed')
print(model.get_accuracy())
print(model2.get_accuracy())

# Получение количества записей по каждому тэгу
# print(dataset['tag'].value_counts())


# print(dataset.iloc[1]['text'])


# for i in range(dataset.shape[0]):


# res_dataset = pandas.DataFrame()
# test_dataset = pandas.DataFrame()
# tags = dataset['tag'].unique().tolist()
# for tag in tags:
#     res_dataset = res_dataset.append(dataset[dataset['tag'] == tag][:50], ignore_index=True)
#     test_dataset = test_dataset.append(dataset[dataset['tag'] == tag][32:], ignore_index=True)

# print(test_dataset['tag'].value_counts())

# model = Pipeline([('vect', CountVectorizer()),
#                                ('tfidf', TfidfTransformer()),
#                                ('clf', LogisticRegression(n_jobs=1,
#                                                           C=1e5,
#                                                           max_iter=100,
#                                                           solver='saga',
#                                                           multi_class='multinomial'
#                                                           ))])
# model.fit(res_dataset['preprocessed'], res_dataset['tag'])
# y_predicted = model.predict(test_dataset['preprocessed'])
# print(accuracy_score(y_predicted, test_dataset['tag']))
# print(classification_report(test_dataset['tag'], y_predicted, target_names=dataset['tag'].unique()))
