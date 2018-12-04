import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting = 3)

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = [0]
for i in range(0,1000):
    
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][0])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] 