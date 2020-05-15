import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import matplotlib.pyplot as plt
train_data=pd.read_csv("train.csv")
test_data=pd.read_csv("test.csv")
print(train_data.columns)

y_train=train_data["SalePrice"]
train_data.drop(columns=["SalePrice"],inplace=True)
desc_train=train_data.describe()

check_train=train_data.isna().any()
check_test=test_data.isna().any()

train_data.shape
test_data.shape

check_train=check_train[check_train==True].index
check_test=check_test[check_test==True].index
check_train
check_test.shape
#  to see which columns are common in both the stuff
common_col=[]
for i in check_test:
    if i in check_train:
        common_col.append(i)

check_train
common_col=np.array(common_col)
common_col

# dealing with the nans
train_data[train_data["LotFrontage"].isna()==True].count()
test_data[test_data["LotFrontage"].isna()==True].count()

train_data.columns


common_col

# seeing the zoning are and replacing here with the median
test_data["MSZoning"].fillna("RL",inplace=True)
# labelencoding here MSZONING

MSZONING=LabelEncoder()
MSZONING.fit(train_data["MSZoning"])
train_data["MSZoning"]=MSZONING.transform(train_data['MSZoning'])

test_data["MSZoning"]=MSZONING.transform(test_data['MSZoning'])
common_col

# filling the stuff with median with lot frontage
train_data["LotFrontage"].fillna(train_data["LotFrontage"].median(),inplace=True)
test_data["LotFrontage"].fillna(test_data["LotFrontage"].median(),inplace=True)
test_data["LotFrontage"].isna().any()
train_data["LotFrontage"].isna().any()

# will cause distrubance thus removing alley function
train_data.drop(columns=["Alley"],inplace=True)
test_data.drop(columns=["Alley"],inplace=True)

check_test
# dealing with utulities columns
train_data['Utilities'].value_counts()
drop_index=train_data[train_data["Utilities"]=='NoSeWa'].index
train_data.drop(drop_index,inplace=True)
y_train.drop(drop_index,inplace=True)
train_data['Utilities'].value_counts()
test_data["Utilities"].fillna("AllPub",inplace=True)

check_train
check_test

# Basement fin type dealt
train_data["BsmtFinType2"].fillna("Unf",inplace=True)

test_data["BsmtFinType2"].fillna("Unf",inplace=True)

bsmtfintype=LabelEncoder()
bsmtfintype.fit(train_data["BsmtFinType2"])
train_data["BsmtFinType2"]= bsmtfintype.transform(train_data["BsmtFinType2"])
test_data["BsmtFinType2"]= bsmtfintype.transform(test_data["BsmtFinType2"])

# bsmtfintype1

test_data['BsmtFinType1'].value_counts()
train_data['BsmtFinType1'].value_counts()

train_data["BsmtFinType1"].fillna("GLQ",inplace=True)
test_data['BsmtFinType1'].fillna("Unf",inplace=True)

check_train=train_data.isna().any()
check_test=test_data.isna().any()
check_train=check_train[check_train==True]
check_test=check_test[check_test==True]

dtypes_see=[]
for i in train_data.columns:
    dtypes_see.append(train_data["{}".format(i)].dtypes)

dtypes_see
np.dtype("int64")

test_data["LotArea"].mean()
from sklearn.impute import SimpleImputer
def do_imputations(data):
    for i in data.columns:
        if data["{}".format(i)].isna().any():
            if data["{}".format(i)].dtype!=np.dtype('int64'):
                data["{}".format(i)]=data["{}".format(i)].fillna(data['{}'.format(i)].value_counts().index[0])
            else:
                data['{}'.format(i)]=data["{}".format(i)].fillna(data["{}".foramt(i)].median())

train_data_copy=train_data
do_imputations(train_data_copy)
test_data_copy=test_data
do_imputations(test_data_copy)
test_data.columns.shape
train_data.columns.shape

train_data_copy.drop(columns=["SalePrice"],inplace=True)
from sklearn.ensemble import RandomForestRegressor
# (test_data_copy.isnull().any()==True).count()


trainer=RandomForestRegressor(n_estimators=1000)

for i in train_data_copy.columns:
    if train_data_copy['{}'.format(i)].dtype==np.dtype("O"):
        print(i)

def label_encoder_stuff(data):
    store_label={}
    for i in data.columns:
        if data['{}'.format(i)].dtype==np.dtype("O"):
            a=LabelEncoder()
            a.fit(data[str(i)])
            data[str(i)]=a.transform(data[str(i)])
            store_label[str(i)]=a
    return store_label

final_train_data=train_data_copy
label_encoders_train_data=label_encoder_stuff(final_train_data)

def encode_test_data(data,label_encodings):
    does_not_encode=[]
    for i in data.columns:
        if i in label_encodings.keys():
            try:
                data[str(i)]=label_encodings[str(i)].transform(data[str(i)])
            except:
                print(str(i))
                does_not_encode.append(i)
    return does_not_encode

final_test_data=test_data_copy

does_not=encode_test_data(final_test_data,label_encoders_train_data)

final_test_data.drop(columns=['Id'],inplace=True)
final_train_data.drop(columns=["Id"],inplace=True)
trying_forest_regerssor=RandomForestRegressor(n_estimators=1000)
trying_forest_regerssor.fit(final_train_data,y_train)

from sklearn.metrics import confusion_matrix

final_test_data.to_csv("new_test.csv")
final_train_data.to_csv("new_train.csv")

y_train_copy=y_train

from sklearn.preprocessing import MinMaxScaler
a=MinMaxScaler()
y_train_copy=a.fit_transform(y_train_copy.to_numpy().reshape((-1,1)))
y_train_copy=pd.Series(y_train_copy.reshape((-1,)))

x_train=pd.read_csv("new_train.csv")
x_test=pd.read_csv("new_test.csv")



