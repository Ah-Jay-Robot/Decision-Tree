# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:18:04 2019

@author: Yugar
"""

# attri_name : attri_value
# 
class Node:
    # actually, the branch is a dict
    # 
    def __init__(self, attri_name = None, attri_value = None, branch = None,
                 label = None, gain = None, is_leaf = False, precision = None):
        self.is_leaf = is_leaf
        # 最优划分
        self.attri_name = attri_name
        self.attri_value = attri_value
        self.branch = branch
        # 所属类别
        self.label = label
        self.gain = gain
        self.precision = precision