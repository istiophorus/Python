import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer
from sklearn import metrics

def silent_remove(arr,val):
    if val in arr:
        arr.remove(val)
        
def plot_corr(df, size=11):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)        
    
df = pd.read_csv("d:/WorkFolder/pima-data/pima-data.csv")
bool_map = {True:1, False:0}
df["diabetes_val"] = df["diabetes"].map(bool_map)

feature_col_names = [n for n in df.columns]
silent_remove(feature_col_names, "diabetes")
silent_remove(feature_col_names, "diabetes_val")
predicted_class_names = ["diabetes_val"]

def test_algo(algo, X_train, X_test, Y_train, Y_test):
    model = algo.fit(X_train, Y_train.ravel())
    
    nb_predict_train = model.predict(X_train)
    score = metrics.accuracy_score(Y_train, nb_predict_train)
    print(score)
    score = metrics.recall_score(Y_train, nb_predict_train)
    print(score)
    
    nb_predict_test = model.predict(X_test)
    score = metrics.accuracy_score(Y_test, nb_predict_test)
    print(score)
    score = metrics.recall_score(Y_test, nb_predict_test)
    print(score)


df.corr(method="kendall")

df.corr(method="spearman")

plot_corr(df)

x = df[feature_col_names].values
y = df[predicted_class_names].values
split_test_size = 0.30

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=split_test_size, random_state=42)

fill_0 = Imputer(missing_values=0, strategy="mean", axis=0)

X_train_clean = fill_0.fit_transform(X_train)
X_test_clean = fill_0.fit_transform(X_test)

from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()

test_algo(nb_model, X_train_clean, X_test_clean, Y_train, Y_test)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42)

test_algo(rf_model, X_train_clean, X_test_clean, Y_train, Y_test)

from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(C=0.7, random_state = 42)

test_algo(lr_model, X_train_clean, X_test_clean, Y_train, Y_test)

from sklearn.linear_model import LogisticRegressionCV
lr_cv_model = LogisticRegressionCV(n_jobs = 1, random_state = 42, Cs = 3, cv = 10, refit = False, class_weight = "balanced")

test_algo(lr_cv_model, X_train_clean, X_test_clean, Y_train, Y_test)

