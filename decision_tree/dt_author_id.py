#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
# Accuracy 0.965870307167
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.tree import DecisionTreeClassifier
from time import time

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
clf = DecisionTreeClassifier(min_samples_split=40)
t0= time()
clf.fit(features_train, labels_train)
print "fitting time:", round(time()-t0, 3), "s"
print "featuers", len(features_train[0])

t1 = time()
print clf.score(features_test, labels_test)
print "fitting time:", round(time()-t1, 3), "s"



#########################################################
### your code goes here ###


#########################################################


