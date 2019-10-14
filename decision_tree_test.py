# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 20:03:14 2019

@author: Yugar
"""

import pandas as pd
import pd_decision_tree
import function

df = pd.read_csv('train.csv')
df.pop('密度')
df.pop('含糖率')
df.pop('编号')
trans_dict = {'是':1,'否':0}
y = [trans_dict[x] for x in df.pop('好瓜')]

root = pd_decision_tree.tree_generate(df, y, function.optimal_partition_CART)

def print_tree(root):
    queue = []
    
    queue.append(root)
    while queue != []:
        node = queue[0]
#        print(node.branch)
        queue.remove(node)
        if node.is_leaf:
#            if node.label == 1:
#                print('好')
#            else:
#                print('坏')
            return
        else:
#            print(node.branch)
#            print(node.gain)
#            print(node.attri_name)
            for k,n in node.branch.items():
                queue.append(n)
#                print(k,n)
                print(node.attri_name, k)
                if node.branch[k].label == 1:
                    print('好')
                elif node.branch[k].label == 0:
                    print('坏')

tf = pd.read_csv('train.csv')
tf.pop('密度')
tf.pop('含糖率')
tf.pop('编号')
yt = [trans_dict[x] for x in tf.pop('好瓜')]


#root.label = 1
#pd_decision_tree.prepruning(df,y,'色泽', root)

#print_tree(root)
data = {'纹理':'清晰', '根蒂':'稍蜷', '色泽':'青绿', '触感': '硬滑'}
res = pd_decision_tree.decision_tree_classfy(root, data)

print('是好瓜吗？')
print(dict(zip([1,0],['是','否']))[res])

