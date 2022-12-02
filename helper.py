import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

import Lab4_helper

from sklearn.utils import resample

def make_tree_rf(X,y,min_split_count=5, prevFeature = "", prevPrevFeature = "", type = "classification"):
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
    bestFeature, rig = Lab4_helper.select_split2(X, y, type=type)
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

def make_trees(x, t, ntrees, type="classification"):
    trees = []
    for _ in range(ntrees):
        #grabbing a random resample of the rows
        x_i, y_i = resample(x, t)
        tree = make_tree_rf(x_i, y_i, type=type)
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
    feature_results = {}
    for trial in range(ntrials):
        X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.25, random_state=trial)
        if alg == "sk_random_forest":
            regressor = RandomForestRegressor(n_estimators=ntrees, min_samples_split=5).fit(X_train,t_train)
            y = regressor.predict(X_test)
        if alg == "my_random_forest":
            trees = make_trees(X_train, t_train, ntrees=ntrees, type="regression")
            rules = make_rules(trees)
            y = make_pred_from_rules(X_test, rules, default=1)
            y = y[0]

        #computing feature importance
        if alg == "sk_random_forest" and importance_metric == "gini":
            importances = regressor.feature_importances_
        elif alg == "sk_random_forest" and importance_metric == "permutation":
            importances = permutation_importance(regressor, X_test, t_test, random_state=trial).importances_mean

        if alg == "sk_random_forest":
            for i in range(len(regressor.feature_names_in_)):
                feature = regressor.feature_names_in_[i]
                importance = importances[i] 
                if feature not in feature_results:
                    feature_results[feature] = [importance]
                else:
                    feature_results[feature].append(importance)
        elif alg == "my_random_forest":
                for tree in trees:
                    gini_importance_from_tree(X_train, t_train, tree, len(X_train), feature_results)

        #since our expected results will always be integers, we might as well round our results
        y = y.round()
        RMSEs.append(np.sqrt(((y-t_test)**2).sum()/len(t_test)))

    average_RMSE = sum(RMSEs) / len(RMSEs)
    if alg == "sk_random_forest":
        feature_results = pd.DataFrame(feature_results).T.mean(axis=1).sort_values(ascending=False)
    elif alg == "my_random_forest":
        feature_results = {key: sum(value) / len(value) for key, value in feature_results.items()}
        feature_results = pd.Series(feature_results).sort_values(ascending=False)

    return average_RMSE, feature_results

def gini(x):
    counts = x.value_counts()
    fracs = counts / len(x)
    ans = 1 - (fracs ** 2).sum()
    return ans

def split_data(x, t, tree):
    feature_name, threshold = list(tree.keys())[0].split("<")
    threshold = float(threshold)

    #Split the data
    x_l = x[x[feature_name] < threshold]
    x_r = x[x[feature_name] >= threshold]
    t_l = t[x[feature_name] < threshold]
    t_r = t[x[feature_name] >= threshold]

    return x_l, x_r, t_l, t_r

def gid(x, t, tree):
    #split the data by the metric in the tree. The node n is the head node of the tree. 
    #Grab the metric in question
    x_l, x_r, t_l, t_r  = split_data(x, t, tree)
    
    #calculate gid
    p_l = len(x_l) / len(x)
    p_r = len(x_r) / len(x)

    gini_n = gini(t)
    gini_l = gini(t_l)
    gini_r = gini(t_r)

    ans = gini_n - (p_l * gini_l + p_r * gini_r)
    return ans
    
#function for recursing through tree and calculating gid at every node:
def gini_importance_from_tree(x, t, tree, n_samples, feature_results):
    if len(x) == 0:
        return 
    if not isinstance(tree, dict):
        return
    feature_name, threshold = list(tree.keys())[0].split("<")
    gid_i  = gid(x, t, tree)
    importance = gid_i * (len(x) / n_samples)
    if feature_name in feature_results:
        feature_results[feature_name].append(importance)
    else:
        feature_results[feature_name] = list([importance])

    #recursing
    subtree = list(tree.values())[0]
    for expected_value, next_tree in subtree.items():
        sub_x = x[(x[feature_name] < float(threshold)) == (expected_value == "True")]
        sub_t = t[(x[feature_name] < float(threshold)) == (expected_value == "True")]
        gini_importance_from_tree(sub_x, sub_t, next_tree, n_samples, feature_results)