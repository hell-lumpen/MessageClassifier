import re
import string

import nltk
import pandas
from nltk.corpus import stopwords
from pymystem3 import Mystem
from tqdm.auto import tqdm

from settings import CONFIG


class MessagePreprocessor:
    __stopwords = [w for w in stopwords.words('russian')
                   if w not in CONFIG['classifier_settings']['exclusionary_stopwords']] \
                  + CONFIG['classifier_settings']['stopwords']

    def __init__(self, filename: str, feature_column_name='text', label_column_name='tag'):
        self.filename = filename
        self.data = pandas.read_csv(filename,
                                    usecols=[feature_column_name, label_column_name])

    def preprocess(self, remove_punc=True, remove_numbers=True, lemmatize=True,
                   remove_stopwords=True) -> pandas.DataFrame:
        self.data['preprocessed'] = [self.preprocess_msg(msg=text,
                                                         remove_punc=remove_punc,
                                                         remove_numbers=remove_numbers,
                                                         lemmatize=lemmatize,
                                                         remove_stopwords=remove_stopwords) for text in
                                     tqdm(self.data['text'])]

        self.data.to_csv(CONFIG['classifier_settings']['prep_data_file'], header=True)

    @staticmethod
    def preprocess_msg(msg: str, remove_punc=True, remove_numbers=True, lemmatize=True,
                       remove_stopwords=True) -> str:
        msg = re.sub(r'\s+', ' ', msg, flags=re.I).lower()
        if remove_punc:
            msg = "".join([ch if ch not in string.punctuation else ' ' for ch in msg])

        if remove_numbers:
            msg = ''.join([i if not i.isdigit() else ' ' for i in msg])

        if remove_stopwords:
            tokenized_msg = nltk.word_tokenize(msg)
            msg = ""
            for token in tokenized_msg:
                if token not in MessagePreprocessor.__stopwords:
                    msg += (token + " ")

        if lemmatize:
            mystem = Mystem()
            msg = mystem.lemmatize(msg)

        return "".join(msg)
