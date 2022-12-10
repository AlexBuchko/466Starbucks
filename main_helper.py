import copy
import json

import numpy as np
import pandas as pd
import random as random

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

#
# All of the code for this next section is just my lab5 code
#

def get_learner(X,y,max_depth=10):
    return DecisionTreeRegressor(max_depth=max_depth).fit(X,y)

def make_prediction(trees,X):
    predictions = []
    tree_predictions = []
    for j in range(len(trees)):
        tree = trees[j]
        tree_predictions.append(tree.predict(X).tolist())
    return np.array(pd.DataFrame(tree_predictions).mean().values.flat)

#my implementation of a boosting algorithm
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

#
# This is now code that I took from lab5 and changed somewhat to get RMSE and permutation based
# feature importances for the data using our boosting algorithm.
#

#
# Our Methods
#

# this is to look at how our output varried from the expected
def get_predict(X, t):
        trial = 1
        X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.25,random_state=trial)
        X_train2, X_val, t_train2, t_val = train_test_split(X_train, t_train, test_size=0.25,random_state=trial)
        trees,train_RMSEs,val_RMSEs = make_trees_boost(X_train2, X_val, t_train2, t_val, max_ntrees=100)
        trees = cut_trees(trees,val_RMSEs)
        y = make_prediction_boost(trees,X_test)
        for i in range(len(y)):
            y[i] = round(y[i])
        return pd.Series(y), pd.Series(t_test)

# we use this to get the RMSE value for our data
def calc_MAE(X, t):
    trials=1
    # this was originally trials = 50, or something like that however this runs unbrearably slow due to the size
    # of the dataset, so get the results in less than a day we dropped this down to 1 trial. 
    mae = []
    for trial in range(trials):
        X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.25,random_state=trial)
        X_train2, X_val, t_train2, t_val = train_test_split(X_train, t_train, test_size=0.25,random_state=trial)
        trees,train_RMSEs,val_RMSEs = make_trees_boost(X_train2, X_val, t_train2, t_val, max_ntrees=100)
        trees = cut_trees(trees,val_RMSEs)
        y = make_prediction_boost(trees,X_test)
        for i in range(len(y)):
            y[i] = round(y[i])
        mae.append((y - t_test).abs().mean())
    return sum(mae)/trials

# this is the method we use for our gradient boosting (permutation based) feature importance and Mean Average Error
def do_feature_importance(X, y):
    importances = {}
    baseMAEave = calc_MAE(X,y)
    #this also prints out our base RMSE so we don't have to do it sepatately
    display(baseMAEave)
    cols = X.columns
    for col in cols:
        X2 = X.drop([col], axis=1)
        temp = calc_MAE(X2, y) - baseMAEave
        importances[col] = temp
        display(col + ": " + str(temp))
    return importances

#
# SKLearn Methods
#

# this method gives us sklearn mean average error
def do_MAE(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=1)
    gbc = GradientBoostingRegressor().fit(X_train, y_train)
    pred = gbc.predict(X_test)
    mae = (pred - y_test).abs().mean()
    return mae

# this is where we get the sklearn feature importances
def get_importances(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=1)
    gbc = GradientBoostingRegressor().fit(X_train, y_train)
    return gbc.feature_importances_

# this is where can see our predictions compared to our expected, this is a sanity check that our
# regressor is running somewhat correctly.
def get_predictions(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=1)
    gbc = GradientBoostingRegressor().fit(X_train, y_train)
    y = gbc.predict(X_test)
    for i in range(len(y)):
        y[i] = round(y[i])
    return pd.Series(y), pd.Series(y_test)


#for getting confusion matrix for analysis:
def check_predictions(expected, actual):
    display(expected.value_counts())
    display(actual.value_counts())
    conf = np.zeros((11,11))
    label = []
    for i in range(11):
        label.append(i)
    for i in range(len(expected)):
        a = int(expected.iloc[i])
        b = int(actual.iloc[i])
        conf[a][b] +=1
    df = pd.DataFrame(data=conf, index=label, columns=label)
    return df
    #correct = 0
    #for i in range(11):
    #    temp = df.iloc[i][i]
    #    display(temp)
    #    correct += temp
    #wrong = len(expected) - correct    
    #return correct, wrong
