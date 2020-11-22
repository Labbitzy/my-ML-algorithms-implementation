# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 12:26:38 2020

@author: Liz
"""
import numpy as np
import math

def SGD(R, Q, P, user_list, item_list, l1, l2, eta = 0.1, threshold = 0.001):
    '''
    
    '''
    err0 = float("inf")
    err = cal_err(user_list, item_list, R, Q, P) 
    while abs(err0-err) > threshold:
        err0 = err
        for (i,u) in zip(item_list,user_list):
            sig = 2*(R[i,u]-np.dot(Q[i,:],P[:,u]))
            Q[i,:] = np.round(Q[i,:]+eta*(sig*P[:,u]-l2*Q[i,:]),4)
            P[:,u] = np.round(P[:,u]+eta*(sig*Q[i,:]-l1*P[:,u]),4)
        err = cal_err(user_list, item_list, R, Q, P)
        print(err)
    return(Q,P)

def build_QP(nitems, nusers, nfactors):
    Q = np.random.rand(nitems,nfactors)
    P = np.random.rand(nfactors, nusers)
    return(Q,P)
    
def cal_err(user_list, item_list, R, Q, P):
    n = len(user_list)
    err = 0
    diff = R-np.dot(Q,P)
    for (i,u) in zip(item_list,user_list):
        err += math.sqrt(diff[i,u]*diff[i,u])/n 
    return(err)
    
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
R = np.zeros((num_items, num_users))
for i in range(len(trainset)):
    R[trainset['item_id'][i]-1,trainset['user_id'][i]-1] = trainset['rating'][i]

user_list = list(trainset['user_id']-1)
item_list = list(trainset['item_id']-1)

ni,nu = R.shape
Q,P = build_QP(nitems = ni, nusers = nu, nfactors = 3)
nQ,nP = SGD(R, Q, P, user_list, item_list, l1 = 0.002,l2 = 0.002,eta = 0.01, threshold = 0.001)
cal_err(user_list, item_list, R, nQ, nP)
Y = np.dot(nQ,nP)

y_true = testset['rating']
y_pred = []
for i in range(len(testset)):
    y_pred.append(Y[testset['item_id'][i]-1, testset['user_id'][i]-1])
    
#report rmse 0.972
from sklearn.metrics import mean_squared_error
rmse_LF = mean_squared_error(y_true, y_pred)
print(rmse_LF)