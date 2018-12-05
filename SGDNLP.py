import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting = 3)

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] 
    review = ' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values 

    
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state = 0)

def sgd():
    # Fitting SVM classifier to the Training set
    from sklearn import linear_model
    clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
    clf.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = clf.predict(X_test)
    
    # Making the confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    return confusion_matrix(y_test, y_pred)