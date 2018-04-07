from nltk import PorterStemmer
from nltk import word_tokenize
from copy import deepcopy

def stemming_tokenizer(text):
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in word_tokenize(text)]

def filter_dataset_nan(x, y):
    docs = deepcopy(x)
    labels = deepcopy(y)

    indices = list()
    for i, doc in enumerate(docs):
        if type(doc) == float:
            indices.append(i)

    return [doc for i, doc in enumerate(docs) if i not in indices], [label for i, label in enumerate(labels) if i not in indices]
