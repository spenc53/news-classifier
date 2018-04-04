from sklearn.cross_validation import train_test_split

from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

import string

from .utils import stemming_tokenizer

# Use LinearSVC and NuSVC. Also figure out why this is getting 25% accuracy

class SVMClassifier:

    def __init__(self):
        pass

    def train(self, x, y):

        classifier = Pipeline([
            ('vectorizer', TfidfVectorizer(tokenizer=stemming_tokenizer,
                                           stop_words=stopwords.words('english') + list(string.punctuation),
                                           min_df = 5)),
            ('classifier', SVC())
        ])

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=33)

        classifier.fit(x_train, y_train)

        print("Accuracy: %s" % classifier.score(x_test, y_test))
        return classifier
