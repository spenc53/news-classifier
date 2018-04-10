import string

import numpy as np
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from classifiers.utils import stemming_tokenizer


class RandomForestClf():

    def __init__(self):
        super().__init__()


    def train(self, X, y):

        classifier = Pipeline([
            ('vectorizer', TfidfVectorizer(tokenizer=stemming_tokenizer,
                                           stop_words=stopwords.words('english') + list(string.punctuation),
                                           min_df=5)),
            ('clf', RandomForestClassifier(n_estimators=275, bootstrap=True, criterion='entropy', min_samples_split=10))
        ])

        # Split data up into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)

        # train the naive bayes classifier
        classifier.fit(X_train, y_train)

        return classifier.score(X_test, y_test)


