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
        self.c = 10
        self.tolerance=1e-3
        self.kernel = 'linear'
        self.model_map = {
                            'default' : SVC(C=self.c, kernel=self.kernel, decision_function_shape='ovr', tol=self.tolerance),
                            'nu' : NuSVC(kernel=self.kernel),
                            'linear' : LinearSVC(C=self.c)
                         }
        self.key = 'default'

    def train(self, x, y):

        classifier = Pipeline([
            ('vectorizer', TfidfVectorizer(tokenizer=stemming_tokenizer,
                                           stop_words=stopwords.words('english') + list(string.punctuation),
                                           min_df = 5)),
            ('classifier', self.model_map[self.key])
        ])

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25)

        classifier.fit(x_train, y_train)

        return classifier.score(x_test, y_test)

class SVMLinearClassifier(SVMClassifier):

    def __init__(self):
        super(SVMLinearClassifier, self).__init__()

        self.key = 'linear'

class SVMNuClassifier(SVMClassifier):

    def __init__(self):
        super(SVMNuClassifier, self).__init__()

        self.key = 'nu'

