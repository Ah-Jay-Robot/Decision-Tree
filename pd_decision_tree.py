# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 19:54:40 2019

@author: Yugar

using data from Zhihua Zhou's book Machine Learning
"""

from collections import Counter
import pandas as pd
from Node import Node
import function
from function import partition_by


def load_test():
    trans_dict = {'是':1,'否':0}
    tf = pd.read_csv('test.csv')
    tf.pop('密度')
    tf.pop('含糖率')
    tf.pop('编号')
    yt = [trans_dict[x] for x in tf.pop('好瓜')]
    
#    print(tf, yt)
    
    return tf, yt

# the input:
#   train set D
#   attri set A

#def intrinsic(values):
#    res = 0.0
#    for i, l in values:
##        print(i / l)
#        res += i / l * math.log2(i / l)
#    return -res
#
## input:
##   gains ->   (the gain, list) , list: the number of this sample
## output:
##   gain_ratio
#def gain_ratio(gains, values):
#    return gains / intrinsic(values)
#
#color = [(6, 17), (6, 17), (5, 17)]
#res = intrinsic(color)
#
#touch = [(5, 17), (12,17)]
#res = gain_ratio(0.109 ,color)
#
#print(res)

# the ds of D : 
# A : unsorted_attri_list []
def tree_generate(D, y, partition_f):
    node = Node()
# ------------------------------- generate leaf
    # all samples of D is in the same class
    #   the target is the same
    # flag represent if the described is true
    if len(set(y)) == 1:
        # mark the class of the node
#        print(D)
        node.label = y[0]
        node.is_leaf = True
        return node

    # mark node as leaf node
    #   the class is the class whose sample number are most
    elif function.is_attri_identical(D) or D.keys() is None:
        _,label = Counter(y).most_common(1)[0]
        node.label = int(label)
        node.is_leaf = True
        return node
# ------------------------------- generate branch
    # a_star is the attribute selected by the optimal partition principle
    
    ### below is 3 ways of implementing the decision tree
    # they are : ID3 / C4.5 / CART
#    a_star, gain = function.optimal_partition_ID3(D,y)
#    a_star, gain = function.optimal_partition_C4_5(D,y)
#    a_star, gain = function.optimal_partition_CART(D,y)
    a_star, gain = partition_f(D,y)
    
#    print(a_star)
    
    # 预剪枝：若分叉后没有增长，则剪枝
    if prepruning(D,y,a_star,node):
        node.is_leaf = True
        return node
    
    # 划分属性的依据
    node.gain = gain
    node.attri_name = a_star
    
    # a is the value of the attri
    node.branch = {}
    for v in set(D[a_star]):
        # select datas where the value of a_star is v
        
        # Dv 为   a_star 属性值为  v  的样本子集
        cols = list(D.keys())
        rows = []
        yi = []
        for i in range(len(D.values)):
            if D[a_star][i] == v:
                rows.append(D.values[i])
                yi.append(y[i])
        Dv = pd.DataFrame(rows, columns = cols)
        Dv.pop(a_star)
#        if v == '稍蜷':
#            print(v)
#            print(D)
#            print(Dv)
#            break
        
        ### 不存在这种情况
#        if Dv is None:5
            # mark node as leaf node
            #   the class is the class whose sample number are most
#            Dv 
#        else:
        node.branch[v] = Node()
#        node.branch[v].attri_value = v
        node.branch[v] = tree_generate(Dv,yi,partition_f)
    return node

# data is a dict
def decision_tree_classfy(root, data):
    node = root
    while not node.is_leaf:
#        print(node.branch.keys())
        if node.attri_name in data.keys() and data[node.attri_name] in node.branch.keys():
            a_value = data[node.attri_name]
            node = node.branch[a_value]
        else:
            print('The attribute', node.attri_name, 'does not have value')
            return
    return node.label


#--------------------------------Prepruning
# Pruning Branch
def prepruning(D, y, k, root):
#    print(D, y)    
    p = 0.0
    # steps :
    #   1. get the P1 from the root
    # find the class whose sample are more
    cnt = 0
    tot = len(D)
    for i in range(tot):
        if y[i] == 1:
            cnt += 1
    p = float(cnt) / tot
    # calcu P
    if p < 0.5:
        root.label = 0
    else:
        root.label = 1
    
    #   2. get the P2 from the branches
    # divide data according to a_star
    
    # branch records which class every value belong to
    branch = dict()
    
    Ds = partition_by(D, y, k)
    for Dv, yi in Ds:
        # 
        pi = 0.0
        cnt = 0
        tot = len(Dv)
        for i in range(tot):
            if yi[i] == 1:
                cnt += 1
        pi += float(cnt) / tot
        # Dv[k][i] is the attri_value
        
#        print(cnt, "/",tot)
#        print(Dv[k].iloc[0], pi)
        
        if pi < 0.5:
            branch[Dv[k].iloc[0]] = 0
        else:
            branch[Dv[k].iloc[0]] = 1
    
    # import test set
    Dt,yt = load_test()
    # calcu P using root
    pb = 0.0
    tot = len(Dt)
    cnt = 0
    for i in range(tot):
        if yt[i] == root.label:
            cnt += 1
    pb = float(cnt) / tot

    
    # calcu P using branch
    pa = 0.0
    cnt = 0
    Ds = partition_by(Dt, yt, k)
    for Dv, yi in Ds:
#        print(list(Dv[k])[0])
#        print(Dv)
        # 
        cnti = 0
        toti = len(Dv)
        for i in range(toti):
#            print(list(Dv[k])[0])
            if yi[i] == branch.setdefault(list(Dv[k])[0], 0):
                cnti += 1
        
        # 输出每个类对应的精度
#        print(cnti, "/",toti)
#        print(Dv[k].iloc[0], cnti / toti)
        
        cnt += cnti
        
    pa = float(cnt) / len(D)

    
#    print(branch)
#    print("pb:",pb)    
#    print("pa:",pa)

        
    # prune after is better than prune before , then prune
    if pa <= pb:
        # prune
        return True
    else:
        return False

# Pclass/Age/Sex
#df = pd.read_csv('train.csv')
#df.pop('PassengerId')
#df.pop('Name')
#df.pop('SibSp')
#df.pop('Ticket')
#df.pop('Fare')
#df.pop('Cabin')
#df.pop('Embarked')
#df.pop('Parch')
#age = pd.cut(df["Age"], np.arange(0, 90, 10))
#df.pop('Age')
#df.insert(2,'Age',age)
#
#y = df['Survived']
#df.pop('Survived')
