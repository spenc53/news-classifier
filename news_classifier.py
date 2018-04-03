import argparse
import os
import requests
from bs4 import BeautifulSoup
# from sklearn.datasets import fetch_20newsgroups
import pandas as pd

from classifiers.naive_classifier import NaiveBayesClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


class NewsClassifier:

    def main(self):
        args = self.parser().parse_args()
        classifier_type = args.C

        classifier = self.get_classifier(classifier_type)

        # IF WE NEED TO RE-GET THE LINKS RUN THE FOLLOWING LINE.
        self.read_links()

        X, Y = self.get_csv_data()
        classifier.train(X, Y)



    def get_classifier(self, classifier_name):
        modelmap = {
            "naive": NaiveBayesClassifier()
            # "svm": SVMClassifier(),
            # "grid": GridSearchClassifier()
        }
        if classifier_name in modelmap:
            return modelmap[classifier_name]
        else:
            raise Exception("Unrecognized model: {}".format(classifier_name))

    def parser(self):
        parser = argparse.ArgumentParser(description='News Articles Classifier Manager')

        parser.add_argument('-C', required=True,
                            choices=['naive', 'svm', 'grid'],
                            help='Classifier')
        return parser

    def get_data(self):
        # Loading the data set - training data.
        twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

        # In[4]:

        # You can check the target names (categories) and some data files by following commands.
        print(twenty_train.target_names)  # prints all the categories

        # In[5]:

        print("\n".join(twenty_train.data[0].split("\n")[:3]))  # prints first line of the first data file

        # In[6]:

        # Extracting features from text files
        from sklearn.feature_extraction.text import CountVectorizer

        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(twenty_train.data)
        print(X_train_counts.shape)

        # In[7]:

        # TF-IDF
        from sklearn.feature_extraction.text import TfidfTransformer

        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        print(X_train_tfidf.shape)

        return X_train_tfidf, twenty_train

    def get_csv_data(self):

        data = pd.read_csv('datasets' + os.path.sep + 'training.csv', encoding='utf-8')  # text in column 1, classifier in column 2.
        numpy_array = data.as_matrix()
        X = numpy_array[:, 1]
        Y = numpy_array[:, 2]

        return X, Y

    def read_links(self):
        csv_data = pd.read_csv('datasets' + os.path.sep + 'links.csv', encoding='utf-8')  # text in column 1, classifier in column 2.
        numpy_array = csv_data.as_matrix()
        links = numpy_array[:, 0]
        classification = numpy_array[:, 1]
        headers = {'User-Agent': 'Mozilla/5.0'}

        article_and_classifications = []


        for i in range(len(links)):
            print(i)
            r = ""
            try:
                r = requests.get(links[i], headers=headers)
            except Exception: 
                print(links[i])
                continue
            data = r.text
            soup = BeautifulSoup(data, 'lxml')
            text = ''
            for p in soup.find_all('p'):
                text = text + '\n' + p.getText()
            article_and_classifications.append([text, classification[i]])

        df = pd.DataFrame(article_and_classifications, columns=["Article", "Classification"])
        df.to_csv('datasets' + os.path.sep + 'training.csv', sep=',')

if __name__ == "__main__":
    NewsClassifier().main()
