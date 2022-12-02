import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

import Lab4_helper

from sklearn.utils import resample

def make_tree_rf(X,y,min_split_count=5, prevFeature = "", prevPrevFeature = ""):
    tree = {}
    differentYVals = y.unique()
    defaultAns = y.value_counts().sort_values(ascending=False).index[0]

    #selecting a subset of the features
    numFeatures = int(np.round_(np.sqrt(X.shape[1])))
    X = X.sample(n = numFeatures, axis="columns")
    
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
    bestFeature, rig = Lab4_helper.select_split2(X, y)
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
        tree[bestFeatureName][str(value).capitalize()] = make_tree_rf(RestOfX, YAtVal, prevFeature=bestFeatureName, prevPrevFeature=prevFeature)
    
    return tree

def make_trees(x, t, ntrees):
    trees = []
    for _ in range(ntrees):
        #grabbing a random resample of the rows
        x_i, y_i = resample(x, t)
        tree = make_tree_rf(x_i, y_i)
        trees.append(tree)

    return trees

def make_rules(trees):
    rules = []
    for tree in trees:
        rule = Lab4_helper.generate_rules(tree)
        rules.append(rule)
    
    return rules

def make_pred_from_rules(x, rules, default = 0):
    predictions = pd.DataFrame()
    for rule in rules:
        #making the prediction from a single truee
        pred = x.apply(lambda x: Lab4_helper.make_prediction(rule,x,default),axis=1)
        #adding it to our dataframe of predictions
        predictions = pd.concat([predictions, pred], axis = 1)

    #we want the prediction to be the most populator prediction from our subset of predictions
    aggregate_prediction = predictions.mode(axis=1)
    return aggregate_prediction

def do_kdd(X, t, ntrials, ntrees, alg, importance_metric="gini"):
    #Helper function for setting up and running a kdd algorithm
    RMSEs =  []
    sk_feature_results = {}
    for trial in range(ntrials):
        X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.25, random_state=trial)
        if alg == "sk_random_forest":
            regressor = RandomForestClassifier(n_estimators=ntrees).fit(X_train,t_train)
            y = regressor.predict(X_test)
        if alg == "my_random_forest":
            trees = make_trees(X_train, t_train, ntrees=ntrees)
            rules = make_rules(trees)
            y = make_pred_from_rules(X_test, rules, default=default)

        #computing feature importance
        if importance_metric == "gini":
            importances = regressor.feature_importances_
        elif importance_metric == "permutation":
            importances = permutation_importance(regressor, X_test, t_test, random_state=trial).importances_mean

        
        for i in range(len(regressor.feature_names_in_)):
            feature = regressor.feature_names_in_[i]
            importance = importances[i] 
            if feature not in sk_feature_results:
                sk_feature_results[feature] = [importance]
            else:
                sk_feature_results[feature].append(importance)

        #since our expected results will always be integers, we might as well round our results
        y = y.round()
        RMSEs.append(np.sqrt(((y-t_test)**2).sum()/len(t_test)))


    importances = pd.DataFrame(sk_feature_results).T.mean(axis=1).sort_values(ascending=False)
    average_RMSE = sum(RMSEs) / len(RMSEs)
    return average_RMSE, importances