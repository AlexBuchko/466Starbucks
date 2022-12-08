import copy
import json

import numpy as np
import pandas as pd
import random as random

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

def get_learner(X,y,max_depth=10):
    return DecisionTreeRegressor(max_depth=max_depth).fit(X,y)

def make_prediction(trees,X):
    predictions = []
    tree_predictions = []
    for j in range(len(trees)):
        tree = trees[j]
        tree_predictions.append(tree.predict(X).tolist())
    return np.array(pd.DataFrame(tree_predictions).mean().values.flat)

def make_trees_boost(Xtrain, Xval, ytrain, yval, max_ntrees=100,max_depth=2):
    trees = []
    F = get_learner(Xtrain, ytrain, max_depth)
    trees.append(F)
    #---------------------------------------------
    ytotal = make_prediction([F], Xtrain)
    ytotal_val = make_prediction([F], Xval)
    #---------------------------------------------
    train_RMSEs = [] # the root mean square errors for the train dataset
    val_RMSEs = [] # the root mean square errors for the validation dataset
    #---------------------------------------------
    ytrain_orig = copy.deepcopy(ytrain)
    yval_orig = copy.deepcopy(yval)
    for i in range(max_ntrees-1):
        y_res = ytrain_orig - ytotal
        y_res_val = yval_orig - ytotal_val
        #do RMSE here
        train_RMSEs.append(np.sqrt(((y_res)**2).sum()/len(y_res)))
        val_RMSEs.append(np.sqrt(((y_res_val)**2).sum()/len(y_res_val)))
        #do rest
        h = get_learner(Xtrain, y_res, max_depth)
        ytotal += make_prediction([h], Xtrain)
        ytotal_val += make_prediction([h], Xval)
        trees.append(h)
    return trees,train_RMSEs,val_RMSEs

def cut_trees(trees,val_RMSEs):
    prevVal = val_RMSEs[0]
    retTrees = [trees[0]]
    for i in range(len(val_RMSEs)-1):
        if(val_RMSEs[i+1] > prevVal):
            return retTrees
        else:
            prevVal = val_RMSEs[i+1]
            retTrees.append(trees[i+1])
    return retTrees

def make_prediction_boost(trees,X):
    tree_predictions = []
    for tree in trees:
        tree_predictions.append(tree.predict(X).tolist())
    return np.array(pd.DataFrame(tree_predictions).sum().values.flat)

def calc_RMSE(X, t):
    trials=1
    RMSE = []
    for trial in range(trials):
        X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.25,random_state=trial)
        X_train2, X_val, t_train2, t_val = train_test_split(X_train, t_train, test_size=0.25,random_state=trial)
        trees,train_RMSEs,val_RMSEs = make_trees_boost(X_train2, X_val, t_train2, t_val, max_ntrees=100)
        trees = cut_trees(trees,val_RMSEs)
        y = make_prediction_boost(trees,X_test)
        for i in range(len(y)):
            y[i] = round(y[i])
        RMSE.append(np.sqrt(((y-t_test)**2).sum()/len(t_test)))
    return sum(RMSE)/trials

def do_feature_importance(X, y):
    importances = {}
    baseRMSEave = do_RMSE(X,y)
    cols = X.columns
    for col in cols:
        X2 = X.drop([col], axis=1)
        temp = do_RMSE(X2, y) - baseRMSEave
        importances[col] = temp
    return importances


def do_RMSE(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=1)
    gbc = GradientBoostingRegressor().fit(X_train, y_train)
    return gbc.score(X_test, y_test)

def get_importances(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=1)
    gbc = GradientBoostingRegressor().fit(X_train, y_train)
    return gbc.feature_importances_

def get_predictions(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=1)
    gbc = GradientBoostingRegressor().fit(X_train, y_train)
    return gbc.predict(X_test), y_test.value_counts()


        