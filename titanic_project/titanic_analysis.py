import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.estimator import DNNClassifier
from tensorflow.feature_column import categorical_column_with_hash_bucket

import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
import os
# files=os.listdir("titanic")
# load csv files
def embarked_dealer(dataset):
    s = []
    q = []
    c = []
    for i in dataset["Embarked"]:
        if(i=="S"):
            s.append(1)
            q.append(0)
            c.append(0)
        elif(i=="C"):
            s.append(0)
            q.append(0)
            c.append(1)
        else:
            s.append(0)
            q.append(1)
            c.append(0)
    value=pd.concat([dataset,pd.DataFrame({"em_s":pd.Series(s),"em_q":pd.Series(q),"em_c":pd.Series(c)})],axis=1)
    value.drop(columns=["Embarked"],inplace=True)
    return value




basepath="titanic"
files=[]
for i in os.listdir(basepath):
    files.append(basepath+"/"+i)
# print(files)
# reading the files

train=pd.read_csv(files[0])
test=pd.read_csv(files[2])
# final labels
labels=train['Survived']
print(train.columns)
# labels=train["Survived"]
# # cleasing data
train.drop(columns=["PassengerId","Survived","Name","Cabin","Ticket"],inplace=True)
test.drop(columns=["PassengerId",'Name',"Cabin","Ticket"],inplace=True)
# print(train.describe())
print(train.columns)
print(test.columns)
# preprocessing the data
# preprocessing the age
train["Age"].fillna(np.mean(train["Age"]),inplace=True)
test["Age"].fillna(np.mean(test["Age"]),inplace=True)
#
print(train.isnull().any())
# making the enteries into binary
train["Sex"]=train["Sex"].apply(lambda x: 1 if x=="female" else 0)
test['Sex']=test["Sex"].apply(lambda x: 1 if x=="female" else 0)
#
# experimental
train.drop(columns=["Fare"],inplace=True)
test.drop(columns=["Fare"],inplace=True)
# dealing with embarked situation
train["Embarked"].fillna('S',inplace=True)
test['Embarked'].fillna("S",inplace=True)
#calling the embarked function
train=embarked_dealer(train)
test=embarked_dealer(test)
train["Age"]=train["Age"]/np.max(train["Age"])
test["Age"]=test["Age"]/np.max(test["Age"])
print(train.columns)
# print("----------------")
# print(test.columns)
# print(train["SibSp"].unique())

# print(train['Sex'].head())

# --------------------------------------------------------
# using the sklearn.ensemble.RandomForestClassifier got almost 80 percent accuracy.
training_x,training_y=(train[0:712],labels[0:712])
validation_x,validation_y=(train[712:],labels[712:])
model=RandomForestClassifier()
model.fit(training_x,training_y)
p=model.score(validation_x,validation_y)
print(p)
#

