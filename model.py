#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 11:43:48 2020

@author: sampoplack
"""


import pandas as pd
import pickle
import numpy as np
df = pd.read_csv("https://raw.githubusercontent.com/hoshangk/machine_learning_model_using_flask_web_framework/master/adult.csv")
import numpy 
from sklearn import preprocessing 

df = df.drop(['fnlwgt', 'educational-num'], axis = 1) 
  
col_names = df.columns 

for c in col_names: 
    df = df.replace("?", numpy.NaN) 
df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))

df.replace(['Divorced', 'Married-AF-spouse',  
              'Married-civ-spouse', 'Married-spouse-absent',  
              'Never-married', 'Separated', 'Widowed'], 
             ['divorced', 'married', 'married', 'married', 
              'not married', 'not married', 'not married'], inplace = True) 
  
category_col =['workclass', 'race', 'education', 'marital-status', 'occupation', 
               'relationship', 'gender', 'native-country', 'income']  
labelEncoder = preprocessing.LabelEncoder() 
  
mapping_dict ={} 
for col in category_col: 
    df[col] = labelEncoder.fit_transform(df[col]) 
  
    le_name_mapping = dict(zip(labelEncoder.classes_, 
                        labelEncoder.transform(labelEncoder.classes_))) 
  
    mapping_dict[col]= le_name_mapping 
print(mapping_dict) 

from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
  
X = df.values[:, 0:12] 
Y = df.values[:, 12] 

X_train, X_test, y_train, y_test = train_test_split( 
           X, Y, test_size = 0.3, random_state = 100) 
  
dt_clf_gini = DecisionTreeClassifier(criterion = "gini", 
                                     random_state = 100, 
                                     max_depth = 5, 
                                     min_samples_leaf = 5) 
  
dt_clf_gini.fit(X_train, y_train) 
y_pred_gini = dt_clf_gini.predict(X_test) 


pickle.dump(dt_clf_gini, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

print(model.predict(X_test))

