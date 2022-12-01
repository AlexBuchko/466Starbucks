import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


def sk_random_forest(X, t, ntrials, ntrees, importance_metric="gini"):
    #Helper function for setting up and running a kdd algorithm
    RMSEs =  []
    sk_feature_results = {}
    for trial in range(ntrials):
        X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.25, random_state=trial)
        classifier = RandomForestClassifier(n_estimators=ntrees).fit(X_train,t_train)

        #computing feature importance
        if importance_metric == "gini":
            importances = classifier.feature_importances_
        elif importance_metric == "permutation":
            importances = permutation_importance(classifier, X_test, t_test, random_state=trial).importances_mean

        for i in range(len(classifier.feature_names_in_)):
            feature = classifier.feature_names_in_[i]
            importance = importances[i] 
            if feature not in sk_feature_results:
                sk_feature_results[feature] = [importance]
            else:
                sk_feature_results[feature].append(importance)

        #computing f1 scores
        y = classifier.predict(X_test)
        RMSEs.append(np.sqrt(((y-t_test)**2).sum()/len(t_test)))


    importances = pd.DataFrame(sk_feature_results).T.mean(axis=1).sort_values(ascending=False)
    average_RMSE = sum(RMSEs) / len(RMSEs)
    return average_RMSE, importances