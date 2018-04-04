from sklearn.cross_validation import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
# from nltk.corpus import punkt

from .utils import stemming_tokenizer


class NaiveBayesClassifier:

    def __init__(self):
        pass

    def train(self, X, y):


        # FUTURE WORK
        #   - Use different classifier
        #       - MultinomialNB(alpha = .05)
        #           - use different alpha levels
        #       - GaussianNB(priors = None)
        #           - get working...
        #       - BernoulliNB
        #           - change priors
        #   - Try different vectorizor
        #       - use default tokenizer function
        #       - change stop_words
        #       - use min_df to say how many times a word must appear to use that word
        classifier = Pipeline([
            ('vectorizer', TfidfVectorizer(tokenizer=stemming_tokenizer,
                                    stop_words=stopwords.words('english') + list(string.punctuation),
                                    min_df = 5)),
            ('classifier', MultinomialNB(alpha=0.05)), # acc of 96% with alpha = .05
            # ('classifier', GaussianNB(priors = None)), # THIS NB DOES NOT WORK
            # ('classifier', BernoulliNB()), # acc of 76.96%, should try changing priors     
        ])
        # Split data up into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 33)

        # print(y_train)

        # train the naive bayes classifier
        classifier.fit(X_train, y_train)

        print("Accuracy: %s" % classifier.score(X_test, y_test))
        return classifier
        # pass

# Really good example can be found here: https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a
# I used this as a guide https://nlpforhackers.io/text-classification/
