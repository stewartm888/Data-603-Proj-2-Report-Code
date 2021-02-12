import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn import svm as hi
import sklearn
import tensorflow as tf
import tensorflow.keras.layers as KL
import inflect
import pandas as pd

typelist = ["CNN","FF","RRN"]

poseepochlist = [5,10,20,50]
poseotmilist = ["Adam","SGD","Adagrad","Adamax"]
digepochlist = [2,5,10,15,20]
digotmilist = ["ADAM","STOC_GRAD","ADAGRAD","RMSPROP"]

img = []

mat = scipy.io.loadmat('pose.mat')
#tmat = scipy.io.loadmat('mnist.mat')
data = mat['pose']
#tdata = mat['mnist']
dim1 = 68
poseotmi_acc = [[56.86,29.90,14.754,51.47],[1.47,1.47,14.22,30.39],[12.25,5.39,3.43,11.76]]

trainpose = []
testpose = []

traindigit = []
testdigit = []


for j in range(0,dim1):
    for i in range(0,13):
        new = data[:,:,i,j]
        if i < 10:
            trainpose.append(new)
            traindigit.append(new)
        elif i <= 13:
            testpose.append(new)
            testdigit.append(new)


trainpose = np.asarray(trainpose)
testpose = np.asarray(testpose)
trainpose_C = np.expand_dims(trainpose, axis = -1)
testpose_C = np.expand_dims(testpose, axis = -1)


labels_train = np.zeros((dim1*10))
labels_test = np.zeros((dim1*3))

epochcount = 20

for i in range(0,dim1):
    labels_train[i*10:i*10+10] = i
    labels_test[i*3:i*3+3] = i


#Mode 1: Feed-forward NN 
############

shapedim = (48,40)
inputs = KL.Input(shape=shapedim)
l = KL.Flatten()(inputs)
l = KL.Dense(512, activation=tf.nn.relu)(l)
outputs = KL.Dense(256,activation=tf.nn.softmax)(l)
model = tf.keras.models.Model(inputs,outputs)
model.summary
model.compile(optimizer = "Adamax", loss="sparse_categorical_crossentropy", metrics =["accuracy"])
model.fit(trainpose,labels_train, epochs =epochcount)#problem
test_loss, test_acc = model.evaluate(testpose, labels_test)


#Mode 2: CNN 
############

inputs_C = KL.Input(shape=(48,40, 1))
poseepoch_acc = [[40.68,39.22,39.21,41.17],[0.49,0.98,0.49,1.47],[5.39,5.39,6.862,9.803]]
c = KL.Conv2D(512, (3,3), padding = "valid", activation = tf.nn.relu)(inputs_C)
m = KL.MaxPool2D((3,3), (3,3))(c)
f = KL.Flatten()(m)
outputs_C = KL.Dense(dim1,activation=tf.nn.softmax)(f)

model_C = tf.keras.models.Model(inputs_C,outputs_C)
model_C.summary
model_C.compile(optimizer = "adam", loss="sparse_categorical_crossentropy", metrics =["accuracy"])
model_C.fit(trainpose_C,labels_train, epochs=epochcount)#problem
test_loss, test_acc = model_C.evaluate(testpose_C, labels_test)


#Mode 3: RNN
############

inputs_RNN = KL.Input(shape=shapedim)
x = KL.SimpleRNN(512, activation="sigmoid")(inputs_RNN)
outputs_RNN = KL.Dense(512, activation="softmax")(x)
model_RNN = tf.keras.models.Model(inputs_RNN, outputs_RNN)
model_RNN.summary()
model_RNN.compile(optimizer = "adam", loss="sparse_categorical_crossentropy", metrics =["acc"])
model_RNN.fit(trainpose,labels_train, epochs=epochcount)#problem
test_loss, test_acc = model_RNN.evaluate(testpose, labels_test)



########################################################
########################################################

# ----------------- MNIST  ----------------- 
########################################################
########################################################

epochlist = [2,5,10,15,20]
otmilist = ["ADAM","STOC_GRAD","ADAGRAD","RMSPROP"]
typelist = ["CNN","FF","RRN"]

#Mode 1: CNN 
############
# load train and test dataset
digepoch_acc =  [[98.2,97.0,92.6,90.01],
[70.1,79.1,85.6,87.1,89.3],[91.2,95.5,97.2,96.1]]
# define cnn model
digotmi_acc = [[],[89.3,85.39,82.39,71.39],
[97.2,95.1,94.9]]
import keras
from keras.datasets import mnist
import keras.backend as k
#load mnist dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data() #everytime loading data won't be so easy :)

import matplotlib.pyplot as plt
fig = plt.figure()
for i in range(9):
  plt.subplot(3,3,i+1)
  plt.tight_layout()
  plt.imshow(X_train[i], cmap='gray', interpolation='none')
  plt.title("Digit: {}".format(y_train[i]))
  plt.xticks([])
  plt.yticks([])
fig

#reshaping
#this assumes our data format
#For 3D data, "channels_last" assumes (conv_dim1, conv_dim2, conv_dim3, channels) while 
#"channels_first" assumes (channels, conv_dim1, conv_dim2, conv_dim3).
img_rows, img_cols = 28, 28
if k.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
#more reshaping
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape) #X_train shape: (60000, 28, 28, 1)

import keras
#set number of categories
num_category = 10
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_category)
y_test = keras.utils.to_categorical(y_test, num_category)

##model building
model = keras.Sequential()
#convolutional layer with rectified linear unit activation
model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
#32 convolution filters used each of size 3x3
#again
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
#64 convolution filters used each of size 3x3
#choose the best features via pooling
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
#randomly turn neurons on and off to improve convergence
model.add(keras.layers.Dropout(0.25))
#flatten since too many dimensions, we only want a classification output
model.add(keras.layers.Flatten())
#fully connected to get all relevant data
model.add(keras.layers.Dense(128, activation='relu'))
#one more dropout for convergence' sake :) 
model.add(keras.layers.Dropout(0.5))
#output a softmax to squash the matrix into output probabilities
model.add(keras.layers.Dense(num_category, activation='softmax'))

#Adaptive learning rate (adaDelta) is a popular form of gradient descent rivaled only by adam and adagrad
#categorical ce since we have multiple classes (10) 
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

batch_size = 128
num_epoch = 2
#model training
model_log = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=num_epoch,
          verbose=1,
          validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0]) #Test loss: 0.0296396646054
print('Test accuracy:', score[1]) #Test accuracy: 0.9904

import os
# plotting the metrics
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(model_log.history['acc'])
plt.plot(model_log.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.subplot(2,1,2)
plt.plot(model_log.history['loss'])
plt.plot(model_log.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.tight_layout()
fig




input("Press for POSE")
for x in typelist:
    print("POSE",x,"\n")
    for y in poseepoch_acc[typelist.index(x)]:      
        print("EPOCH {} ACC%  -- {}".format(poseepochlist[poseepoch_acc[typelist.index(x)].index(y)],y))
    print("\n")
    for y in poseotmi_acc[typelist.index(x)]:       
        print("OPTIMIZER {} ACC%  -- {}".format(poseotmilist[poseotmi_acc[typelist.index(x)].index(y)],y))
    print("\n------------\n")
input("Press for MNIST")
for x in typelist:
    print("MNIST", x,"\n")
    for y in poseepoch_acc[typelist.index(x)]:      
        print("EPOCH {} ACC%  -- {}".format(poseepochlist[poseepoch_acc[typelist.index(x)].index(y)],y))
    print("\n")
    for y in poseotmi_acc[typelist.index(x)]:       
        print("OPTIMIZER {} ACC%  -- {}".format(poseotmilist[poseotmi_acc[typelist.index(x)].index(y)],y))
    print("\n------------\n")
