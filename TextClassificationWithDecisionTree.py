# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 15:30:30 2018

@author: Terry
"""

#import numpy as np
#import matplotlib.pyplot as plt 
import pandas as pd 

#learning function
def training(positive_reviews, negative_reviews):
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

def checking(toCheck, weights, sumOf, num):
    for bla in toCheck.split():
        if bla in weights.keys():
            sumOf[num] += weights[bla]
            #print(bla, weights[bla])
    return;

#importing Dataset - first step befor the next command- must set a working directory
dataset = pd.read_json('Musical_Instruments_5.json', lines=True)

#splitting the dataset into the Trainin set and Test set
from sklearn.cross_validation import train_test_split
train, test = train_test_split(dataset, test_size=0.2, random_state = 42)
#splitting test into devSet and testSet
testSet, devSet = train_test_split(test, test_size=0.1, random_state = 42)

sumOfArray = [0.0]
weightOf = []
overallPredict = 0

#creating a matrix of feature - undependent and dependent 
# X - undependent features text
# Y - dependent feature on X - 1-5 review 
 #[lines, coulomns]
devSet = devSet.iloc[:, [2,3]].values

#result = {'score': 0, 'oneErr': 0, 'twoErr': 0, 'threeErr': 0, 'fourErr': 0}
result = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
for trueOverall, stringRev in devSet:
    #decision tree
    #splitting to positive = class of 1-3 and negative = class of 3-5 review 
    positive_reviews = train.loc[(train['overall'] == 1) | (train['overall'] == 2) | (train['overall'] == 3)]
    negative_reviews = train.loc[(train['overall'] == 3) | (train['overall'] == 4) | (train['overall'] == 5)]
    weightOf.append(training(positive_reviews, negative_reviews))
    for num, weig in zip(range(1), weightOf):
        checking(stringRev, weig, sumOfArray, num)  
    #print(sumOfArray[0], "sum")  
    if sumOfArray[0]<0 :
        #print("3 4 5 class")
        #splitting to positive = class of 3-4 and negative = class of 4-5 review  
        positive_reviews = train.loc[(train['overall'] == 3) | (train['overall'] == 4)]
        negative_reviews = train.loc[(train['overall'] == 4) | (train['overall'] == 5)]
        sumOfArray[0] = 0.0 
        weightOf = []
        weightOf.append(training(positive_reviews, negative_reviews))
        for num, weig in zip(range(1), weightOf):
            checking(stringRev, weig, sumOfArray, num)
        #print(sumOfArray[0], "sum")  
        if sumOfArray[0]<0 :
            #print("4 5 class")
            positive_reviews = train.loc[train['overall'] == 4]
            negative_reviews = train.loc[train['overall'] == 5]
            sumOfArray[0] = 0.0 
            weightOf = []
            weightOf.append(training(positive_reviews, negative_reviews))
            for num, weig in zip(range(1), weightOf):
                checking(stringRev, weig, sumOfArray, num)
            #print(sumOfArray[0], "sum")
            if sumOfArray[0]<0 :
                #print("overall prediction is 5")
                overallPredict = 5
            else:
                #print("overall prediction is 4")
                overallPredict = 4
        else:
            #print("3 4 class")
            positive_reviews = train.loc[train['overall'] == 3]
            negative_reviews = train.loc[train['overall'] == 4]
            sumOfArray[0] = 0.0 
            weightOf = []
            weightOf.append(training(positive_reviews, negative_reviews))
            for num, weig in zip(range(1), weightOf):
                checking(stringRev, weig, sumOfArray, num)
            #print(sumOfArray[0], "sum")
            if sumOfArray[0]<0 :
                #print("overall prediction is 4")
                overallPredict = 4
            else:
                #print("overall prediction is 3")
                overallPredict = 3
    else:
        #print("1 2 3 class")
        #splitting to positive = class of 1-2 and negative = class of 2-3 review  
        positive_reviews = train.loc[(train['overall'] == 1) | (train['overall'] == 2)]
        negative_reviews = train.loc[(train['overall'] == 2) | (train['overall'] == 3)]
        sumOfArray[0] = 0.0 
        weightOf = []
        weightOf.append(training(positive_reviews, negative_reviews))
        for num, weig in zip(range(1), weightOf):
            checking(stringRev, weig, sumOfArray, num)
        #print(sumOfArray[0], "sum")
        if sumOfArray[0]<0 :
            #print("2 3 class")
            positive_reviews = train.loc[train['overall'] == 2]
            negative_reviews = train.loc[train['overall'] == 3]
            sumOfArray[0] = 0.0 
            weightOf = []
            weightOf.append(training(positive_reviews, negative_reviews))
            for num, weig in zip(range(1), weightOf):
                checking(stringRev, weig, sumOfArray, num)
            #print(sumOfArray[0], "sum")
            if sumOfArray[0]<0 :
                #print("overall prediction is 3")
                overallPredict = 3
            else:
                #print("overall prediction is 2")
                overallPredict = 2
        else:
            #print("1 2 class")
            positive_reviews = train.loc[train['overall'] == 1]
            negative_reviews = train.loc[train['overall'] == 2]
            sumOfArray[0] = 0.0 
            weightOf = []
            weightOf.append(training(positive_reviews, negative_reviews))
            for num, weig in zip(range(1), weightOf):
                checking(stringRev, weig, sumOfArray, num)
            #print(sumOfArray[0], "sum")
            if sumOfArray[0]<0 :
                #print("overall prediction is 2")
                overallPredict = 2
            else:
                #print("overall prediction is 1")
                overallPredict = 1
                
    print("overallPredict = ", overallPredict , "---->",trueOverall)
    result[abs(trueOverall - overallPredict)] += 1
    
print("result = ", result)