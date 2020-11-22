# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 10:01:12 2020

@author: Liz
"""
import numpy as np
import heapq
from math import sqrt 
from sklearn.metrics import mean_squared_error

class similarity_func:
    def __init__(self, arr1, arr2):
        self.arr1 = arr1
        self.arr2 = arr2
    def Jaccard(arr1, arr2):
        list1 = np.where(arr1 != 0)
        list2 = np.where(arr2 != 0)
        intersection_len = len(set(list1).intersection(list2))
        unionset_len = len(set(list1 + list2))
        jaccard = intersection_len/unionset_len
        return(jaccard)
    
    def cosine(arr1, arr2):
        return(sum(arr1*arr2)/(sqrt(sum(arr1*arr1))*sqrt(sum(arr2*arr2))))
    
    def Pearson(arr1,arr2):
        Ex = np.mean(arr1)
        Ey = np.mean(arr2)
        Exy = np.mean(arr1*arr2)
        Ex2 = np.mean(arr1*arr1)
        Ey2 = np.mean(arr2*arr2)
        return((Exy-Ex*Ey)/(sqrt(Ex2-Ex*Ex)*sqrt(Ey2-Ey*Ey)))

def CF_item(X,user_index,item_index,k = 3,sim = 'Pearson'):
    '''
    input X matrix user-item rating col:user, row:item
    row_index, item_index
    sim: similarity index
    '''
    length_X = len(X)
    if sum(X[item_index]) == 0:
        result = np.mean(X)
    else:
        sim_method = getattr(similarity_func, sim)
        sim_measure = [sim_method(X[item_index],X[i]) for i in range(length_X)]
        ind = heapq.nlargest(k+1, range(len(sim_measure)), key=sim_measure.__getitem__)
        if item_index in ind:
            ind.remove(item_index)
        else:
            ind.remove(ind[0])
        weights = np.array(sorted(sim_measure,reverse=True)[1:1+k])
        if user_index >= np.shape(X)[1]:
            result = np.mean(X)
        else:
            result = sum(X[ind,user_index]*weights)/sum(weights)
    return(result)    
    
def CF_user(X,user_index,item_index,k = 3,sim = 'Pearson'):
    '''
    input X matrix user-item rating col:user, row:item
    row_index, item_index
    sim: similarity index
    '''
    length_X = len(X)
    if sum(X[user_index]) == 0:
        result = 0
    else:
        sim_method = getattr(similarity_func, sim)
        sim_measure = [sim_method(X[user_index],X[i]) for i in range(length_X)]
        ind = heapq.nlargest(k+1, range(len(sim_measure)), key=sim_measure.__getitem__)
        if item_index in ind:
            ind.remove(item_index)
        else:
            ind.remove(ind[0])
        weights = np.array(sorted(sim_measure,reverse=True)[1:1+k])
        result = sum(X[ind,item_index]*weights)/sum(weights)
    return(result)    
    
X = np.array([[1,0,3,0,0,5,0,0,5,0,4,0],
              [0,0,5,4,0,0,4,0,0,2,1,3],
              [2,4,0,1,2,0,3,0,4,3,5,0],
              [0,2,4,0,5,0,0,4,0,0,2,0],
              [0,0,4,3,4,2,0,0,0,0,2,5],
              [1,0,3,0,3,0,0,2,0,0,4,0]])
CF(X,2,2,k=2)

#load dataset
import os
import pandas as pd

def read_data_ml100k(data_dir,filename):
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(os.path.join(data_dir, filename), '\t', names=names,
                       engine='python')
    return(data)

data_dir = 'ml-100k'
trainset = read_data_ml100k(data_dir,'ua.base')
testset  = read_data_ml100k(data_dir,'ua.test')

#build user-item matrix
num_users = max(trainset['user_id'])
num_items = max(trainset['item_id'])
train_X = np.zeros((num_items, num_users))
for i in range(len(trainset)):
    train_X[trainset['item_id'][i]-1,trainset['user_id'][i]-1] = trainset['rating'][i]

testset['user_id'] = testset['user_id']-1
testset['item_id'] = testset['item_id']-1

#item-item RMSE: 8.073
y_pred1 = []
for i in range(len(testset)):
    y_pred1.append(CF_item(train_X, testset['user_id'][i], testset['item_id'][i]))

y_true = testset['rating']
rmse_CF1 = mean_squared_error(y_true, y_pred1)
print(rmse_CF1)

#user-user RMSE: 7.418
train_X = np.zeros((num_users,num_items))
for i in range(len(trainset)):
    train_X[trainset['user_id'][i]-1,trainset['item_id'][i]-1] = trainset['rating'][i]
y_pred2 = []
for i in range(len(testset)):
    y_pred2.append(CF_user(train_X, testset['user_id'][i], testset['item_id'][i]))

rmse_CF2 = mean_squared_error(y_true, y_pred2) 
print(rmse_CF2)   











