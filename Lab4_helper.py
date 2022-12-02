import copy
import json

import numpy as np
import pandas as pd
import math


def entropy(y):
    e = None
    # YOUR SOLUTION HERE
    numObservations = y.size
    frequencies = y.value_counts() / numObservations
    e = -1 * sum([freq * math.log(freq, 2) for freq in frequencies])    
    return e

def gain(y,x):
    g = 0
    # YOUR SOLUTION HERE
    possibleValues = x.unique()
    weightedEntropies = []
    for value in possibleValues:
        xAtVal = x.loc[x == value]
        yAtVal = y.loc[x == value]
        unweightedEntropy = entropy(yAtVal)
        weight = xAtVal.size / x.size
        weightedEntropies.append(weight * unweightedEntropy)
        
    g = sum(weightedEntropies)

    return entropy(y) - g

def gain_ratio(y,x):
    # YOUR SOLUTION HERE
    g = gain(y, x)
    return g/entropy(y)

def select_split(X,y):
    col = None
    gr = None
    # YOUR SOLUTION HERE
    gainRatios = X.aggregate(lambda col: gain_ratio(y, col), axis = 0).sort_values(ascending=False)
    col = gainRatios.index[0]
    gr = gainRatios.iloc[0]
    return col,gr

def make_tree(X,y, prevVal = 0):
    tree = {}
    # Your solution here
    differentYVals = y.unique()
    defaultAns = y.value_counts().sort_values(ascending=False).index[0]
    if(differentYVals.size) == 1:
        return differentYVals[0]
    if(X.shape[0] == 0 or X.shape[1] == 0):
        return defaultAns
    if(X.drop_duplicates().shape[0] == 1):
        return defaultAns
    #then we're in the recursive case
    #pick the field with the highest IG
    bestFeature, rig = select_split(X, y)
    tree[bestFeature] = {}
    possibleFeatureValues = X[bestFeature].unique()
    for value in possibleFeatureValues:
        XAtVal = X.loc[X[bestFeature] == value]
        YAtVal = y.loc[X[bestFeature] == value]        
        RestOfX = XAtVal.drop(columns=bestFeature)
        tree[bestFeature][value] = make_tree(RestOfX, YAtVal, prevVal=value)
    
    return tree

# if you want to print like me :)
def print_tree(tree):
    mytree = copy.deepcopy(tree)
    def fix_keys(tree):
        if type(tree) != dict:
            return tree #int(tree)
        new_tree = {}
        for key in list(tree.keys()):
            if type(key) == np.int64:
                new_tree[int(key)] = tree[key]
            else:
                new_tree[key] = tree[key]
        for key in new_tree.keys():
            new_tree[key] = fix_keys(new_tree[key])
        return new_tree
    mytree = fix_keys(mytree)
    print(json.dumps(mytree, indent=4, sort_keys=True))

def generate_rules(tree):
    rules = []

    def dfs(tree, path):
        if not isinstance(tree, dict):
            path.append(tree)
            rules.append(path)
            return
        nodeName = list(tree.keys())[0]
        node = list(tree.values())[0]
        edges = node.keys()
        for edge in edges:
            newStep = (nodeName, edge)
            curPath = path.copy()
            curPath.append(newStep)
            dfs(node[edge], curPath)
    
    dfs(tree, [])

    return rules

def split_col(x, y):
    x2 = list(x.unique())
    save_x = x.copy()
    x2.sort()
    splits = []
    bestGain = -1
    bestCol = None
    for i in range(0, len(x2)-1):
        splits.append((x2[i] + x2[i+1]) / 2)
    for split in splits:
        x = x.apply(lambda x: True if x < split else False)
        g = gain_ratio(y, x)
        if g > bestGain:
            bestGain, bestCol = g, x.copy().rename(f"{x.name}<{split}0")
            
        x = save_x.copy()
    return bestGain, bestCol

def select_split2(X,y):
    # YOUR SOLUTION HERE
    splitCols = list(map(lambda col: split_col(col[1], y), X.items()))
    splitCols.sort(key=lambda x: x[0], reverse=True)
    bestGr, splitCol = splitCols[0]
    
    return splitCol, bestGr

def make_tree2(X,y,min_split_count=5, prevFeature = "", prevPrevFeature = ""):
    tree = {}
    # Your solution here
    differentYVals = y.unique()
    defaultAns = y.value_counts().sort_values(ascending=False).index[0]
    if(differentYVals.size) == 1:
        return defaultAns
    if(X.shape[0] < min_split_count):
        return defaultAns
    if(X.shape[1] == 0):
        return defaultAns
    if(X.drop_duplicates().shape[0] == 1):
        return defaultAns
    #then we're in the recursive case
    #pick the field with the highest IG
    bestFeature, rig = select_split2(X, y)
    if rig <= 0.001:
        return defaultAns
    
    #replacing the continous column with its split counterpart
    bestFeatureName = bestFeature.name
    originalColumnName = bestFeatureName.split("<")[0]
    X = X.drop(columns=[originalColumnName])
    X[bestFeatureName] = bestFeature

    #setting up tree
    tree[bestFeatureName] = {}
    possibleFeatureValues = [True, False]
    
    #recursing
    for value in possibleFeatureValues:
        XAtVal = X.loc[X[bestFeatureName] == value]
        YAtVal = y.loc[X[bestFeatureName] == value]     
        RestOfX = XAtVal.drop(columns=bestFeatureName)
        tree[bestFeatureName][str(value).capitalize()] = make_tree2(RestOfX, YAtVal, prevFeature=bestFeatureName, prevPrevFeature=prevFeature)
    
    return tree

def make_prediction(rules,x,default):
    # Your solution here
    for rule in rules:
        for clause in rule:
            #if we hit the value
            if not isinstance(clause, tuple):
                return clause
            feature, expectedValue = clause
            if "<" in feature:
                featureName, threshold = feature.split("<")
                
                test =(x[featureName] < float(threshold)) == (expectedValue == "True")
            else:
                test = x[feature] == expectedValue

            if test == False:
                break
    return default
