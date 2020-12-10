# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 21:06:24 2020

@author: Liz
"""

import os
import numpy as np
import pickle as pkl
import random
from gensim.models import Word2Vec
os.chdir('C:\\Users\\labbi\\OneDrive\\Liz\\Brandeis\\computer system\\HW4\\data')

#load data
DATASET = 'cora'
NAMES = ['x', 'y', 'tx', 'ty', 'allx', 'ally','graph']
objects = {}
for name in NAMES:
    data = pkl.load(open("ind.{}.{}".format(DATASET, name), 'rb'), encoding='latin1')
    objects[name] = data
    
G = objects['graph']
y_train = objects['ally']
y_test = objects['ty']

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index
test_idx = parse_index_file("ind.{}.test.index".format(DATASET))

#node2vec
def probtx(t,x,v,edges_dict,p,q):
    if t == x:
        prob = 1/p
    elif t in edges_dict[x]:
        prob = 1
    else:
        prob = 1/q
    return(prob)

def node2vecWalk(G,start_node,length,p,q):
    walk = [start_node]
    while len(walk) < length:
        curr = walk[-1]
        N_curr = G[curr]
        if len(N_curr) > 0:
            s = AliasSample(G, walk,p,q)
            walk.append(s)
        else:
            break      
    walk = [str(i) for i in walk]
    return(walk)
    
def AliasSample(edges_dict, walk,p,q):
    curr = walk[-1]
    neighbors = edges_dict[curr]
    if len(walk) == 1:
        nextnode = random.choices(neighbors)[0]
    else:
        prev = walk[-2]
        prob_weight = tuple([probtx(prev,x,curr,edges_dict,p,q) for x in neighbors])
        nextnode = random.choices(neighbors, weights=prob_weight, k=1)[0]
    return(nextnode)
    
def learn_features(G,d,r,l,p,q,k=10):
    '''
    Graph G = (V, E, W), Dimensions d, Walks per node r, Walk length l, 
    Context size(window size) k, Return p, In-out
    '''
    walks = []
    for i in range(r):
        for node in G.keys():
            walk = node2vecWalk(G,node,l,p,q)
            walks.append(walk)
    #walks = [map(str, walk) for walk in walks]
    model = Word2Vec(walks, size=d, window=k, min_count=0, sg=1, workers=8, iter=1)
    model.wv.save_word2vec_format("word2vec.model")
    feature = model.wv
    return(feature)

dim = 128
Graph_f = learn_features(G=G,d=dim,r=5,l=100,p=0.5,q=2,k=5)

#data
X_test = np.empty((0,dim), int)
for idx in test_idx:
    arr = Graph_f[str(idx)].reshape(1,dim)
    X_test = np.append(X_test,arr,axis = 0)
def Diff(li1, li2):
    return (list(list(set(li1)-set(li2))))
train_idx = Diff(list(range(2708)),test_idx)
X_train = np.empty((0,dim), int)
for idx in train_idx:
    arr = Graph_f[str(idx)].reshape(1,dim)
    X_train = np.append(X_train,arr,axis = 0)
    
#classification accury:0.806
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
print(neigh.score(X_test, y_test))
