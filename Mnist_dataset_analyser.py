import tensorflow as tf
import pandas as pd
import numpy as np
import keras
import pydot
import graphviz
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,Conv2D,MaxPool2D,Input,BatchNormalization,Flatten
from keras.utils import plot_model
from keras.models import Model
from IPython.display import display,Image

import matplotlib.pyplot as plt
print("tensorflow version " + tf.__version__)


def generate_model(model_with_input_layer,conv=True,*args,**kwargs):
    """

    :param model_with_input_layer: return model with layers
    :param conv: if true then cnn returned unless a ann would be returned
    :return: A model (used in composite layer)
    """
    if (conv==True):
        model=Conv2D(filters=32,kernel_size=(3,3),padding="same",activation="relu")(model_with_input_layer)
        model = Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu")(model)
        model=BatchNormalization()(model)
        model = Conv2D(filters=32, kernel_size=(2, 2), padding="same", activation="relu")(model)
        model=MaxPool2D(strides=2)(model)
        model = Conv2D(filters=32, kernel_size=(2, 2), padding="same", activation="relu")(model)
        model = Conv2D(filters=32, kernel_size=(2, 2), padding="same", activation="relu")(model)
        model=Flatten()(model)
        model=Dense(32,activation="relu")(model)
        model=Dense(12,activation="relu")(model)
        output=Dense(10,activation="softmax")(model)
        return output
    else:
        model=keras.layers.Reshape((28*28*128,))(model_with_input_layer)
        model=Dense(128,activation="relu")(model)
        model=Dense(128,activation="relu")(model)
        model = Dense(64, activation="relu")(model)
        model = Dense(32, activation="relu")(model)
        model=Dense(32,activation="relu") (model)
        output=Dense(10,activation='softmax')(model)
        return output
    return


# data import from mnist dataset
(X_train,Y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()
print(X_train.shape)
# showing data
for i in range(9):
    plt.subplot(331+i)
    plt.title("{}".format(Y_train[i]))
    plt.imshow(X_train[i],cmap="gray")


# for more variation in data
X_train=X_train.reshape(X_train.shape[0],28,28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1)
#
print(X_train.shape)
datagenerator=ImageDataGenerator(featurewise_std_normalization=True,rotation_range=50,zca_whitening=False)
datagenerator.fit(X_train)

# visuliasation after data augementation
# using datagenerator
for x,y in datagenerator.flow(X_train,Y_train,batch_size=9):
    for i in range(9):
        plt.subplot(441+i)
        plt.title(y[i])
        plt.imshow(x[i].reshape(28,28),cmap="gray")
    break
plt.show()

X_train=X_train/255
x_test=x_test/255

Y_train=keras.utils.to_categorical(Y_train,num_classes=10)
y_test=keras.utils.to_categorical(y_test,num_classes=10)

# trying a cnn based composite model architecture
input_layer=Input(shape=(28,28,1))
model1=Dense(128,activation="relu")(input_layer)
model2=Dense(128,activation="relu")(input_layer)
model1=generate_model(model1,conv=True)
model2=generate_model(model2,conv=False)
# model1
# model2
# merg
# ing the two models to make one
model_merged=Model(inputs=input_layer,outputs=[model1,model2])
plot_model(model_merged,to_file="this_is_the_merged_model.png",show_layer_names=True)
# use this only for scripts and not for the ipython display here
display(img=Image("this_is_the_merged_model.png"))
# model.summary()


#printing the summary of model.
print(model_merged.summary())

model_merged.compile(optimizer="sgd",loss="categorical_crossentropy",metrics=["accuracy"])
# for train and test data
# there are two outputs and one input thus there is a list of y_train here which does that
# without usind the image augementation
model_merged.fit(x=X_train,y=[Y_train,Y_train],epochs=10)


# with using data augemenation
epochs=10
for l in range(10):
    counter=0
    for x_data,y_data in datagenerator.flow(X_train,Y_train,batch_size=1000):
        model_merged.fit(x=x_data,y=[y_data,y_data])
        counter+=1
        if(counter>15):
            break

