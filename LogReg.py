import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer

from tqdm import tqdm

if __name__ == '__main__':
    
    # Load the data and drop NaN texts from train
    train = pd.read_csv('./fake_news/train.csv').dropna(subset='text')
    test = pd.read_csv('./fake_news/test.csv')
    labels = pd.read_csv('./fake_news/labels.csv')

    # I don't believe empty news :D
    test_idx = ~test.text.isna()
    test.loc[~test_idx, 'prediction'] = 1

    stemmer = PorterStemmer()

    # Retrieve stopwords from all of the available languages into one set
    stop_words = set(sum([stopwords.words(language) for language in stopwords.fileids()], []))

    def clean(text):
        words = word_tokenize(text)
        words = [stemmer.stem(w.lower()) for w in words if  # Stem and lower
                not w in stop_words  # Not a stop word
                and w.isalpha()]  # Only contains letters

        return " ".join(words)


    # Clean the data and calculate Tf-Idfs with filtering very frequent and infrequent words
    train_vectorizer = TfidfVectorizer(min_df=0.005, max_df=0.6)
    train_y = train.label
    train_X = train.text.progress_apply(clean)
    train_X = train_vectorizer.fit_transform(train_X)

    test_X = test.loc[test_idx].text.progress_apply(clean)
    test_X = train_vectorizer.transform(test_X)
    
    # Fit the log. reg.
    model = LogisticRegression(C=0.01, penalty='l2').fit(train_X, train_y)

    test.loc[test_idx, 'prediction'] = model.predict(test_X)

    test_y = labels.label
    test_pred = test.prediction.astype('int')

    print('Test Accuracy:', accuracy_score(test_y, test_pred))
    print('Test f1 score:', f1_score(test_y, test_pred))
    conf_matrix = confusion_matrix(test_y, test_pred) / len(test_y)


    # Same, save the results
    results = {
        'test_accuracy': accuracy_score(test_y, test_pred),
        'test_f1': f1_score(test_y, test_pred),
        'test_confusion_matrix': conf_matrix.tolist()
    }

    with open('results_logreg.json', 'w') as f:
        json.dump(results, f)





