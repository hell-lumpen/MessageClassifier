import re
import string

import nltk
from pandas import DataFrame
import pandas
from nltk.corpus import stopwords
from pymystem3 import Mystem
from tqdm.auto import tqdm
from nltk.stem.snowball import SnowballStemmer

from settings import CONFIG


class MessagePreprocessor:
    __stopwords = [w for w in stopwords.words('russian')
                   if w not in CONFIG['preprocessor_settings']['exclusionary_stopwords']] \
                  + CONFIG['preprocessor_settings']['stopwords']
    __stemmer = SnowballStemmer('russian')

    def __init__(self):
        pass

    def preprocess_file(self, filename=CONFIG['preprocessor_settings']['raw_data_file'],
                        remove_punc=True, remove_numbers=True, lemmatize=True,
                        remove_stopwords=True):
        dataset = pandas.read_csv(filename)
        dataset['preprocessed'] = [self.preprocess_msg(msg=text,
                                                       remove_punc=remove_punc,
                                                       remove_numbers=remove_numbers,
                                                       lemmatize=lemmatize,
                                                       remove_stopwords=remove_stopwords) for text in
                                   tqdm(dataset['text'])]

        dataset.to_csv(CONFIG['preprocessor_settings']['prep_data_file'], header=True)

    def preprocess_dataset(self, dataset: DataFrame, source_label='text', destination_label='preprocessed',
                           remove_punc=True, remove_numbers=True,
                           lemmatize=True, remove_stopwords=True) -> DataFrame:
        res_dataset = DataFrame()
        res_dataset[destination_label] = [self.preprocess_msg(msg=text,
                                                              remove_punc=remove_punc,
                                                              remove_numbers=remove_numbers,
                                                              lemmatize=lemmatize,
                                                              remove_stopwords=remove_stopwords) for text in
                                          tqdm(dataset[source_label])]
        return res_dataset

    @staticmethod
    def preprocess_msg(msg: str, remove_punc=True, remove_numbers=True, lemmatize=True,
                       remove_stopwords=True) -> str:
        msg = re.sub(r'\s+', ' ', msg, flags=re.I).lower()
        if remove_punc:
            msg = "".join([ch if ch not in string.punctuation + '«»' else ' ' for ch in msg])

        if remove_numbers:
            msg = ''.join([i if not i.isdigit() else ' ' for i in msg])

        if remove_stopwords:
            tokenized_msg = nltk.word_tokenize(msg)
            msg = ""
            for token in tokenized_msg:
                if not lemmatize:
                    token = MessagePreprocessor.__stemmer.stem(token)
                if token not in MessagePreprocessor.__stopwords:
                    msg += (token + " ")

        if lemmatize:
            mystem = Mystem()
            msg = mystem.lemmatize(msg)

        filtered_tokens = []
        if remove_stopwords:
            for token in msg:
                if token not in MessagePreprocessor.__stopwords:
                    filtered_tokens.append(token)
        return " ".join(filtered_tokens)[:-2]
