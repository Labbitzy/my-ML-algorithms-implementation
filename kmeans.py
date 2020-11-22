# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 16:47:18 2020

@author: Liz
"""

import pandas as pd
import numpy as np
import numpy_indexed as npi
import random
import time
from sklearn.metrics.cluster import normalized_mutual_info_score
random.seed(814)
#load iris data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris = pd.read_csv(url, names=col_names)

X = np.array(iris.iloc[:,:4])
y = np.array(iris[['class']]).ravel()

def d(x,centroid):
    arr_diff = centroid-x
    diff = np.sum(arr_diff**2, axis = 1)
    return(diff)

#(a)Random    
def random_initials(X, k):
    initials = X[np.random.choice(X.shape[0], k, replace=False)]
    return(initials)

#(b)K-means++  
def Kmeans_plusplus(X,k):
    points_choice = random.sample(range(X.shape[0]), 1)
    while len(points_choice) < k:
        dist = []
        for i in range(X.shape[0]):
            distance = d(X[i],X[points_choice])
            dist.append(min(distance))
        sum_dist = sum(dist)
        prob = [dist[i]/sum_dist for i in range(len(dist))]
        cum_prob = 0
        x = random.uniform(0,1)
        for ind, prob in zip(range(len(dist)), prob):
            cum_prob = cum_prob + prob
            if x < cum_prob:
                break
        if ind not in points_choice:
            points_choice.append(ind)
    initials = X[points_choice]
    return(initials)    

#(c)Global Kmeans
def Global_Kmeans(X,k):
    initials = np.mean(X,axis = 0).reshape((1,X.shape[1]))
    ObjV = float('inf')
    while initials.shape[0] < k:
        for i in range(X.shape[0]):
            centroid = np.concatenate((initials,X[i].reshape((1,X.shape[1]))))
            cluster,centroid_new,ObjV_new = Easy_Kmeans(X,centroid.shape[0],centroid)
            if ObjV_new < ObjV:
                initials_new = centroid_new
                ObjV = ObjV_new
        initials = initials_new
    return(initials)
cen = Global_Kmeans(X,3)
    
def assign_cluster(X,centroid):
    cluster = []
    objV = 0
    for i in range(X.shape[0]):
        arr_diff = d(X[i],centroid)
        cluster.append(np.argmin(arr_diff))
        objV=objV+np.amin(arr_diff)
    return(cluster,objV)

def update_centroid(X, cluster, k):
    new_centroid = npi.group_by(cluster).mean(X)[1]
    return(new_centroid)

def initials_methods(X, k, method):
    '''
    get initialization centroid for given initialization
    input: X, data metrics
           k, number of centroids
    output: initial centroid
    '''
    if method == 'random':
        initials = random_initials(X,k)
    elif method == 'kmeans plusplus':
        initials = Kmeans_plusplus(X,k)
    elif method == 'global kmeans':
        initials = Global_Kmeans(X,k)
    else: print('Please enter correct initialization method(random/kmeans plusplus/global kmeans)')
    return(initials)

def Easy_Kmeans(X,k,centroid):
    best_objV = float('inf')
    objV = float('inf')
    new_objV = 10000000000
    while new_objV < objV:
        objV = new_objV
        cluster, new_objV = assign_cluster(X,centroid)
        centroid = update_centroid(X, cluster, 3)
    if new_objV < best_objV:
        best_cluster = cluster
        best_objV = new_objV
        best_centroid = centroid
    return(best_cluster,best_centroid,best_objV)
    
def Lloyd_Kmeans(X,k,ini_method,n_init=10, Maxiter=300,threshold=0.001,earlystop=False):
    if ini_method == 'global kmeans':
        #global kmeans is deterministic
        n_init = 1
    best_objV = float('inf')
    seed = 814
    for i in range(n_init):
        random.seed(seed)
        centroid = initials_methods(X, k, ini_method)
        for j in range(Maxiter):
            objV = float('inf')
            new_objV = 100000
            if earlystop:
                if objV - new_objV >= threshold:
                    objV = new_objV
                    cluster, new_objV = assign_cluster(X,centroid)
                    centroid = update_centroid(X, cluster, 3)
                else:
                    break
            else:
                if new_objV < objV:
                    objV = new_objV
                    cluster, new_objV = assign_cluster(X,centroid)
                    centroid = update_centroid(X, cluster, 3)
                elif new_objV == objV:
                    break
            if new_objV < best_objV:
                best_cluster = cluster
                best_objV = new_objV
                best_centroid = centroid
        seed+=1
    return(best_cluster,best_centroid,best_objV)
    
#my KMeans
start = time.time()
cluster,centroid,ObjV = Lloyd_Kmeans(X,3,'global kmeans',threshold=0.001,earlystop=False)
end = time.time()
print(end - start)
normalized_mutual_info_score(y, cluster,average_method='arithmetic')

#sklearn KMeans
from sklearn.cluster import KMeans
start = time.time()
kmeans = KMeans(n_clusters=3, random_state=814).fit(X)
cluster_sklearn = kmeans.labels_
end = time.time()
print(end - start)
normalized_mutual_info_score(y, cluster_sklearn,average_method='arithmetic')
