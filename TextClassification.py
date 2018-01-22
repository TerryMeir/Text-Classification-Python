# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 12:00:07 2018

@author: Terry
"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

#importing Dataset - first step befor the next command- must set a working directory
dataset = pd.read_json('Musical_Instruments_5.json', lines=True)

#splitting to negative and positive review
fiveOverallReview = dataset.loc[dataset['overall'] == 5]
complementFiveReview = dataset.loc[dataset['overall'] != 5]

#extract text only
positive_reviews = fiveOverallReview.iloc[:, 3].values
negative_reviews = complementFiveReview.iloc[:, 3].values

weights = {}
pos = []
weights['__bias__'] = 0.0
 
training = [(pr,  1.0) for pr in positive_reviews] + \
           [(nr, -1.0) for nr in negative_reviews]
 
for _ in range(10):          
  for example, label in training:
    dot = weights['__bias__']
    for word in example.split():
      dot += weights.get(word, 0.0)
      
  #  print(example, label)
     
    dot = 1.0 if dot >= 0 else -1.0
     
    if dot * label < 0:
      weights['__bias__'] += label
      for word in example.split():
        weights[word] = weights.get(word, 0.0) + label
 
    
    
for example, label in training:
  dot = weights['__bias__']
  for word in example.split():
    dot += weights.get(word, 0.0)
'''   
for word in weights:
    print (word, weights[word])
'''
sumOf5 = 0.0
sumOf1234 = 0.0
sumOf4 = 0.0
sumOf1235 = 0.0
sumOf3 = 0.0
sumOf1245 = 0.0
sumOf2 = 0.0
sumOf1345 = 0.0
sumOf1 = 0.0
sumOf2345 = 0.0

toCheck = "cable is not good"

for bla in toCheck.split():
    if bla in weights.keys():
        sumOf5 += weights[bla]
        print(bla, weights[bla])
 
    #print(word, dot)
  #print( example, label, 1.0 if dot >= 0 else -1.0)

print(sumOf5, "sum of 5")  
      

"""
#creating a matrix of feature - undependent and dependent 
# X - undependent features e.g name, text, product 
# Y - dependent feature on X - good\bad review 
X = dataset.iloc[:, 3].values #[lines, coulomns] the meaning of : is to select all and :-1 will select all without the lest coulumn 
Y = dataset.iloc[:, 2].values 

#splitting the dataset into the Trainin set and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train , y_test = train_test_split(X, Y, test_size=0.2, random_state = 42)

"""