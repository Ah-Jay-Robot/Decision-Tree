# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:39:49 2019

@author: Yugar
"""
import pandas as pd
import numpy as np
from math import log

global e 
e = 1e6
#X is the value of each attribute, y is the target vaule within {0, 1}
# (569,30)
#X = X[:,:3]# select the first 3 features


# √
# see if the target of all samples
#    in D is the same
# eg.
# D = [([1,2,3],1),([1,2,5],0),([1,2,7],0)]
# print(is_identical(D)) -> True
# if all attri in D is identical
def is_attri_identical(D):
    for v in D.keys():
        if len(set(D[v])) != 1:
            return False
    return True


#--------------------------------ID3
def gain(D, y, attri_name):
    l = len(D.values)
#    attri_name = '色泽'

    gain = 0.0
    cnt = [0,0]
    for i in range(len(D.values)):
        if y[i] == 0:
            cnt[0] += 1
        else:
            cnt[1] += 1
    for i in range(len(cnt)):
        # 可免去判断条件
        if cnt[i] != 0:
            prob = float(cnt[i]) / l
            gain -= prob * log(prob,2)
#    print(gain)
    
# 计算 entropy    
    counts = D[attri_name].value_counts()
    values = counts.keys()
#    print(values)
            
    ent = 0.0
    for v in values:
        cnt = [0,0]
        for i in range(len(D.values)):
            if D[attri_name][i] == v:
#                print(D[attri_name][i])
#                print(y[i])
                if y[i] == 0:
                    cnt[0] += 1
                else:
                    cnt[1] += 1
        for i in range(len(cnt)):
            # 可免去
            if cnt[i] != 0:
                prob = float(cnt[i]) / counts[v]
                ent -= float(counts[v]) / l * prob * log(prob,2)
    gain -= ent
#    print(gain)
    return gain

def optimal_partition_ID3(D,y):
    key = D.keys()[0]
    optimal = gain(D,y,key)
    for k in D.keys()[1:]:
        g = gain(D,y,k)
        if g > optimal:
            key = k
            optimal = g
#        print(g,k)
    return key,optimal

#--------------------------------C4.5
def optimal_partition_C4_5(D,y):
    if len(D.keys()) == 1:
        return D.keys()[0], 0
    
    mean = 0.0
    gains = {}
    for k in D.keys():
        gains[k] = gain(D,y,k)
        mean += gains[k]
    # 平均水平
    mean /= len(D.keys())
    
    gains_above = {}
    for k,v in gains.items():
        # weed out the BE
        if v >= mean - e:
            gains_above[k] = v
           
    gain_ratio = []
    l = len(D.values)
    for k,v in gains_above.items():
        Ds = partition_by(D, y, k)
        IV = 0.0
        for Dv,_ in Ds:
            prob = float(len(Dv.values)) / l
            IV -= prob * log(prob, 2)
        gain_ratio.append(v / IV)
        
    # get the a_star
    gain_ratio_max = max(gain_ratio)
    index = gain_ratio.index(gain_ratio_max)
    a_star = list(gains_above)[index]
    
    return a_star, gain_ratio_max


# classify the D,y according to a
def partition_by(D, y, a):
    # 所有属性值
    a_values = set(D[a])
#    print(a_values)
    res = []
    for a_value in a_values:
        Dv = pd.DataFrame([], columns = D.keys())
        yi = []
        values = []
        for i in range(len(D)):
            if D[a][i] == a_value:
                values.append(D.iloc[i])
                yi.append(y[i])
        Dv = Dv.append(values)
#        Dv.pop(a)
        res.append((Dv,yi))
    return res

#---------------------------------CART
def optimal_partition_CART(D,y):
    if len(D.keys()) == 1:
        return D.keys()[0], 0
    
    Ginis = []
    l = len(D.values)
    for k in D.keys():
        Gini = 1.0
        Ds = partition_by(D, y, k)
        for Dv,_ in Ds:
            prob = float(len(Dv.values)) / l
            Gini -= prob * prob
        Ginis.append(Gini)
        
    # pick the max from Ginis
    #   get the a_star
    Gini_max = max(Ginis)
    index = Ginis.index(Gini_max)
    a_star = list(D.keys())[index]
    
    return a_star, Gini_max
    

if __name__ == '__main__':
    df = pd.read_csv('train.csv')
    
#    values = np.array(df.values)
#    keys = np.array(df.keys())
    trans_dict = {'是':1,'否':0}
    y = [trans_dict[x] for x in df.pop('好瓜')]
    
    df.pop('密度')
    df.pop('含糖率')
    df.pop('编号')
#    print(df.keys())
    
#    a_star, g = optimal_partition(df,y)
    a_star, g = optimal_partition_C4_5(df, y)
#    res = partition_by(df, y, '色泽')
    print(a_star)
#    set(df[a_star])
#    cols = list(df.keys())
#    rows = []
#    for i in range(len(df.values)):
#        if df[a_star][i] == '清晰':
#            rows.append(df.values[i])
#    Dv = pd.DataFrame(rows, columns = cols)
#    Dv.pop('纹理')
    