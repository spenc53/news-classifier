from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

class NaiveBayesClassifier:

    def __init__(self):
        pass

    def train(self, X, y):

        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(X)
        # X_train_counts.shape


        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

        clf = MultinomialNB().fit(X_train_tfidf, y)
        # X_train_tfidf.shape



        # Define how the traning will be done
        # # classifier = Pipeline([
        # #     ('vectorizer', TfidfVectorizer()),
        # #     ('classifier', MultinomialNB()),
        # # ])

        # # X = v.fit_transform(X.values.astype('U'))

        # # print(X)

        # classifier = Pipeline([
        #     ('vectorizer', TfidfVectorizer(tokenizer=stemming_tokenizer,
        #                             stop_words=stopwords.words('english') + list(string.punctuation))),
        #     ('classifier', MultinomialNB(alpha=0.05)),
        # ])

        # v = TfidfVectorizer(decode_error='replace', encoding='utf-8')
        # X = v.fit_transform(X.astype(str))
        # y = v.fit_transform(y.astype(str))

        # X = np.nan_to_num(X)
        # y = np.nan_to_num(y)

        # # Split data up into train and test
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 33)

        # print(y_train)

        # # train the naive bayes classifier
        # classifier.fit(X_train, y_train)

        # print("Accuracy: %s" % classifier.score(X_test, y_test))
        # return classifier
        # pass

# Really good example can be found here: https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a
