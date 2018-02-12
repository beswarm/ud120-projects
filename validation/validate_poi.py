#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from mylib import fit_and_predict

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
sort_keys = '../tools/python2_lesson13_keys.pkl'
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

clf = DecisionTreeClassifier(random_state=42)
acc = fit_and_predict(clf, features, features, labels, labels)
print("Overfitted {0:.3}".format(acc))

clf = DecisionTreeClassifier(random_state=42)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30, random_state=42, shuffle=False)
acc = fit_and_predict(clf, X_train, X_test, y_train, y_test)
print("Accuracy of test {0:.3}".format(acc))

### it's all yours from here forward!  


