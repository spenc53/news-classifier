# News Article Classifier

## Before Using

To install of the required dependencies, make sure you run

```
pip install -r requirements.txt
```

## Usage

In order to run the classifier, you have to specify which classification algorithm to use. 
The possibilities will include Naive Bayes (naive), Subject Vector Machines (svm)
and Grid Search (grip).

Example:

```
python3 news_classifier -C {naive | svm | grip}
```

