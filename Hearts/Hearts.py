#!/usr/bin/env python
# coding: utf-8

# In[77]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# In[53]:


dataSetFileName = "d:/_py_/heart/processed.hungarian.data"


# In[54]:


names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]


# In[55]:


df = pd.read_csv(dataSetFileName, header = None, names = names)


# In[56]:


def find_missing(dff, missingValue):
    result = []
    for c in dff.columns:
        cc = len(dff[dff[c] == missingValue])
        if cc > 0:
            result.append(c)
    return result


# In[57]:


def set_dominant(dff, colName, missingValue):
    dominantValue = dff[dff[colName] != missingValue][colName].value_counts().index[0]
    dff[colName].mask(dff[colName] == missingValue, dominantValue, inplace=True)
    return dominantValue


# In[58]:


def set_domminant_in_all_missing(dff, missingValue):
    cols = find_missing(dff, missingValue)
    for col in cols:
        print(col)
        set_dominant(dff, col, missingValue)


# In[59]:


def set_mean(dff, colName, missingValue):
    meanValue = dff[dff[colName] != missingValue][colName].mean()
    dff[colName].mask(dff[colName] == missingValue, meanValue, inplace=True)
    return meanValue


# In[61]:


set_domminant_in_all_missing(df, '?')


# In[64]:


x = df.iloc[:,0:-1].to_numpy()


# In[68]:


y = df.iloc[:,-1:].to_numpy()


# In[69]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
y_train_rav = y_train.ravel()


# In[78]:


models = [LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier(), SVC()]


# In[79]:


def fit_and_predict_with_model(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train_rav)
    y_pred = model.predict(x_test)    
    acc = accuracy_score(y_test, y_pred)
    return acc


# In[84]:


for model in models:
    print(model.__class__)
    acc = fit_and_predict_with_model(model, x_train, x_test, y_train, y_test)
    print(acc)


# In[ ]:




