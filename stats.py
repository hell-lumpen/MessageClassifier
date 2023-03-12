import nltk
import pandas


class Stats:
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
    def get_stat_by_tag(data: pandas.Series, tags=None | list):
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
