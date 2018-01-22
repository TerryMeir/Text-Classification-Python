# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 17:42:59 2018

@author: Terry
"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

#learning function
def training(posGroup):
    #splitting to negative and positive review
    positive_reviews = train.loc[train['overall'] == posGroup]
    negative_reviews = train.loc[train['overall'] != posGroup]
    
    #extract text only
    positive_reviews = positive_reviews.iloc[:, 3].values
    negative_reviews = negative_reviews.iloc[:, 3].values
    
    weights = {}
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
    
    return weights;

#importing Dataset - first step befor the next command- must set a working directory
dataset = pd.read_json('Musical_Instruments_5.json', lines=True)

#splitting the dataset into the Trainin set and Test set
from sklearn.cross_validation import train_test_split
train, test = train_test_split(dataset, test_size=0.2, random_state = 42)

weightOf = []
for num in range(1,6):
    weightOf.append(training(num))

stringRev = "This guitar sounds awesome and stays in tune very well. I am now a fan of Takamine. Their is nothing bad I can say about this guitar and I would take this guitar anywhere. This is my 4th guitar I have bought and is by far my favorite guitar. I have 2 electric guitars and now 2 acoustic, this being the second acoustic. I wanted a cheap guitar I could take anywhere and not have to worry about it being scratched up, except I got the guitar and looks too nice to let get beat up. People are complaining about the guitar not having a glossy finish. It wasn't supposed to be glossy to give it a richer look. If this guitar was glossy it wouldn't be as nice sounds stupid but it is true. You have to see the guitar face to face to understand what I mean. This guitar I would recommend to anyone. I am trying to get my friends to buy one so can all have a acoustic guitar, lol. Thank You!!!!!!"

def checking(toCheck, weights, sumOf, num):
    for bla in toCheck.split():
        if bla in weights.keys():
            sumOf[num] += weights[bla]
            #print(bla, weights[bla])
    return;

sumOfArray = [0.0, 0.0, 0.0, 0.0, 0.0]

for num, weig in zip(range(5), weightOf):
    checking(stringRev, weig, sumOfArray, num)
   
print(sumOfArray[4], "sum of 5")  
print(sumOfArray[3], "sum of 4")  
print(sumOfArray[2], "sum of 3")  
print(sumOfArray[1], "sum of 2")  
print(sumOfArray[0], "sum of 1")  

result = max(sumOfArray)
print ("max is: ", result )
print("Index is: ", sumOfArray.index(result) )