import MessagePreprocessor
import nltk
import pandas
from wordcloud import WordCloud
from matplotlib import pyplot as plt


class Stats:
    @staticmethod
    def __str_corpus(corpus):
        str_corpus = ''
        for i in corpus:
            str_corpus += ' ' + i
        str_corpus = str_corpus.strip()
        return str_corpus
    @staticmethod
    def __get_wordCloud(corpus):
        wordCloud = WordCloud(background_color='white',
                              width=1000,
                              height=1000,
                              max_words=200,
                              random_state=42
                              ).generate(Stats.__str_corpus(corpus))
        return wordCloud


    @staticmethod
    def __tokenize_data(data: pandas.DataFrame, column_name: str):
        text = data[column_name].tolist()
        tokens = []
        for t in text:
            tokenized_data = nltk.word_tokenize(t)
            for td in tokenized_data:
                tokens.append(td)
        return tokens

    @staticmethod
    def get_word_cloud_plot(data: pandas.DataFrame, column_name: str):
        word_cloud_plot = Stats.__get_wordCloud(Stats.__tokenize_data(data, column_name))
        fig = plt.figure(figsize=(30, 30))
        plt.subplot(1, 2, 1)
        plt.imshow(word_cloud_plot)
        plt.axis('off')

        plt.subplot(1, 2, 1)
        plt.savefig('data/raw_word_cloud.png')
        plt.show()

    @staticmethod
    def get_all_stat(data: pandas.Series):
        stats: dict[str, int] = {}
        text = data.tolist()
        for txt in text:
            tokens = nltk.word_tokenize(txt)
            for token in tokens:
                if token in stats:
                    stats[token] += 1
                else:
                    stats[token] = 1
        return pandas.Series(dict(sorted(stats.items(), reverse=True,
                           key=lambda item: item[1])))

    @staticmethod
    def get_stat_by_tag(data: pandas.Series, tags=None):
        pass

    @staticmethod
    def get_word_stat(data: pandas.Series):
        stats: dict[str, int] = {}
        messages = data.tolist()
        for msg in messages:
            tokens = nltk.word_tokenize(msg)
            for token in tokens:
                if token in stats:
                    stats[token] += 1
                else:
                    stats[token] = 1
        return pandas.Series(dict(sorted(stats.items(), reverse=True,
                                         key=lambda item: item[1])))
