import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting = 3)

def printf(cm):
    TP = cm[1][1]
    FN = cm[0][1]
    FP = cm[1][0]
    TN = cm[0][0]
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1_score = 2 * Precision * Recall / (Precision + Recall)
    print('accuracy =', Accuracy, ' ,precision =',Precision,' ,Recall =', Recall, ' F1Score ' ,F1_score)

#cleaning 
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

import KNNNLP
%run KNNNLP.py
printf(cm)

import decisiontreeNLP
%run decisionTreeNLP.py
printf(cm)

import NaivesBayesNLP
%run NaivesBayesNLP.py
printf(cm)

import SGDNLP
%run KNNNLP.py
printf(cm)

import KNNNLP
%run KNNNLP.py
printf(cm)

import KNNNLP
%run KNNNLP.py
printf(cm)
