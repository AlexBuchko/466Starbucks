import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier



def sk_random_forest(X, t, ntrials, ntrees):
    #Helper function for setting up and running a kdd algorithm
    RMSEs =  []
    sk_feature_results = {}
    for trial in range(ntrials):
        X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.25, random_state=trial)
        classifier = RandomForestClassifier(n_estimators=ntrees).fit(X_train,t_train)

        #computing feature importance
        for i in range(len(classifier.feature_names_in_)):
            feature = classifier.feature_names_in_[i]
            importance = classifier.feature_importances_[i]
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