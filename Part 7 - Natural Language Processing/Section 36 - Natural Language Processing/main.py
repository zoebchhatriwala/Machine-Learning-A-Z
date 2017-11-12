# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning The Texts

# Downloading Nltk stopwords files
#nltk.download('stopwords')

import re   
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def cleanText(review):
    review = re.sub('[^a-zA-Z$]', ' ', review)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    temp_review = review
    review = []
    for word in temp_review:
        if word not in set(stopwords.words('english')):
            review.append(ps.stem(word))
    review = ' '.join(review)
    return review

# Natural language Processing

# Corpus
corpus = []

# Cleaning
for i in range(0, len(dataset)):
    corpus.append(cleanText(dataset['Review'][i]))

# Creating Bags of Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('Results are with Accuracy of ' + str(((cm[0][0]+cm[1][1])/len(y_test))*100) +'%')


# For Input
single_review = cleanText('I would love to come here again')
single_x = cv.transform([single_review]).toarray()
single_y_pred = classifier.predict(single_x)
if single_y_pred[0] == 1:
    print('Thank you for your positive review')
else:
    print('Sorry, For your inconvenience we will try to improve.')





