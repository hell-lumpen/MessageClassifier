from MessagePreprocessor import MessagePreprocessor
from settings import CONFIG


def main():
    msg_processor = MessagePreprocessor(CONFIG['classifier_settings']['row_data_file'])
    msg_processor.preprocess()


if __name__ == "__main__":
    # nltk.download('punkt')
    # nltk.download('stopwords')
    main()
