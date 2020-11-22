# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 00:06:41 2020

@author: Liz
"""
import numpy as np
import random

def d(x,y):
    arr_diff = y-x
    diff = np.sum(arr_diff**2, axis = 1)
    return(diff)
    
def min_k(diff,k):
    index = []
    uni_diff = sorted(list(set(diff)))
    loc = -1
    while len(index) < k:
        loc = loc+1
        min_index = [i for i in range(len(diff)) if diff[i] == uni_diff[loc]]
        if len(index)+len(min_index)<=k:
            add_index = min_index
        else:
            num = k - len(index)
            add_index = random.sample(min_index, k=num)
        index.extend(add_index)
    return(index)
    
def neighhbor(x, X_train, k):
    diff = []
    for i in range(X_train.shape[0]):
        diff.append(d(x,X_train[i]))
    neighhbor_index = min_k(diff,k)
    return(neighhbor_index)
    
def neighbor_label(index,y_train):
    neighhbor_labels = y_train[index]
    return(neighhbor_labels)

def majority_vote(neighhbor_labels):
    counts = dict()
    for i in neighhbor_labels:
        counts[i] = counts.get(i, 0) + 1
    labels = [key for key,val in counts.items() if val == max(counts.values())]
    if len(labels) > 1:
        label = random.choice(labels)
    else:
        label = labels[0]
    return(label)
    
def KNN(X_train, y_train, X_test, k):
    y_test = []
    for i in range(X_test.shape[0]):
        x = X_test[i]
        xn_indexs = neighhbor(x, X_train, k)
        xn_labels = neighbor_label(xn_indexs,y_train)
        xn_labels = list(xn_labels.flatten())
        x_label   = majority_vote(xn_labels)
        y_test.append(x_label)
    return(y_test)
        