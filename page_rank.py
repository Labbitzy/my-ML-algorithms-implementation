# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 00:38:15 2020

@author: Liz
"""
import numpy as np

#'A':['B'] means A->B
edges_dict = {'A':['B','C','D'],'B':['A','D'],'C':['A'],'D':['B','C','E']}

def build_trans(edges_dict, beta):
    key = list(edges_dict.keys())
    value = []
    for i in edges_dict.values():
        value.extend(i)
    keys = sorted(list(set(key+value)))
    N=len(keys)
    M = [ [0]*N for i in range(N) ]
    for a,b in [(keys.index(a), keys.index(b)) for a, row in edges_dict.items() for b in row]:
        M[b][a] = 1/len(list(edges_dict.values())[a])
    M = np.array(M)
    teleport = np.array([1/N]*N*N).reshape((N,N))    
    trans = beta*M+(1-beta)*teleport
    return(trans)

def page_rank(N, trans_matrix):
    rank = 0
    new_rank = np.array([1/N]*N)
    while all(rank != new_rank):
        rank = new_rank
        new_rank = np.dot(trans_matrix,rank)
    return(new_rank)
    
M = build_trans(edges_dict, beta=0.8) 
print(page_rank(N=4, trans_matrix=M))
