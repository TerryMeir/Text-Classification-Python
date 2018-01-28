# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 18:41:09 2018

@author: Terry
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

#importing Dataset 
dataset = pd.read_json('Musical_Instruments_5.json', lines=True, )

#[lines, coulomns]
dataset.drop(['asin','helpful', 'reviewTime', 'reviewerID', 'reviewerName', 'summary', 'unixReviewTime'], axis=1, inplace=True)

#Cleaning the text
import re
#removing all marks from text (e.g !@#$%^&:"....), keeping noly letters
#sub() 
#first arg - what we whant to drop from the test. I will use the nagetive way of what i dont want to drop from the test --> using the ^ which is not.
#secound arg - the replace carrecter, defult is non and then words can stik together
#third arg - working string to change
review = re.sub('[^a-zA-Z]', ' ', dataset['reviewText'][0])
#lowecass for all letters
review = review.lower()
#removing non relevant words as this, a, the... the words list is 'stopwords' downloaded from internet
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
review = review.split()
review = [word for word in review if not word in set(stopwords.words('english'))]   #using set to execute faster cuz list is slower
#stemming words - taking the root of the words - e.g love loved loving will be love
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
review = [ps.stem(word) for word in review]
#after cleaning process I'll convert the list of words to string again
review = ' '.join(review)
