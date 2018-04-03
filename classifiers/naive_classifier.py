from sklearn.cross_validation import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

import string
from nltk.stem import PorterStemmer
from nltk import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
# from nltk.corpus import punkt

import numpy as np

def stemming_tokenizer(text):
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in word_tokenize(text)]


class NaiveBayesClassifier:

    def __init__(self):
        pass

    def train(self, X, y):

        # Define how the traning will be done
        # classifier = Pipeline([
        #     ('vectorizer', TfidfVectorizer()),
        #     ('classifier', MultinomialNB()),
        # ])

        # X = v.fit_transform(X.values.astype('U'))

        # print(X)

        classifier = Pipeline([
            ('vectorizer', TfidfVectorizer(tokenizer=stemming_tokenizer,
                                    stop_words=stopwords.words('english') + list(string.punctuation))),
            ('classifier', MultinomialNB(alpha=0.05)),
        ])

        # v = TfidfVectorizer(decode_error='replace', encoding='utf-8')
        # X = v.fit_transform(X.astype(str))
        # y = v.fit_transform(y.ast)

        # X = np.nan_to_num(X)
        # y = np.nan_to_num(y)

        # Split data up into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 33)

        print(y_train)

        # train the naive bayes classifier
        classifier.fit(X_train, y_train)

        print("Accuracy: %s" % classifier.score(X_test, y_test))
        return classifier
        # pass

# Really good example can be found here: https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a
