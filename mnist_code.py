import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn import svm as hi
import sklearn
import tensorflow as tf
import tensorflow.keras.layers as KL
import inflect


### Pose
########################################################

img = []

mat = scipy.io.loadmat('pose.mat')
data = mat['pose']
dim1 = 68


trainpose = []
testpose = []

for j in range(0,dim1):
    for i in range(0,13):
        new = data[:,:,i,j]
        if i < 10:
            trainpose.append(new)
        elif i <= 13:
            testpose.append(new)


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

########################################
### Architecture #1 Feed-forward NN ####
########################################

#model Feed-Forward
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
print("FF Architecture's Acc%: {} for {} epochs".format(test_acc,epochcount))


############################
### Architecture #2 CNN ####
############################

#CNN
inputs_C = KL.Input(shape=(48,40, 1))
c = KL.Conv2D(512, (3,3), padding = "valid", activation = tf.nn.relu)(inputs_C)
m = KL.MaxPool2D((3,3), (3,3))(c)
f = KL.Flatten()(m)
outputs_C = KL.Dense(dim1,activation=tf.nn.softmax)(f)
model_C = tf.keras.models.Model(inputs_C,outputs_C)
model_C.summary
model_C.compile(optimizer = "adam", loss="sparse_categorical_crossentropy", metrics =["accuracy"])
model_C.fit(trainpose_C,labels_train, epochs=epochcount)#problem
test_loss, test_acc = model_C.evaluate(testpose_C, labels_test)
print("CNN Architecture's Acc%: {} for {} epochs".format(test_acc,epochcount))

############################
### Architecture #3 RNN ####
############################

#RNN
inputs_RNN = KL.Input(shape=shapedim)
x = KL.SimpleRNN(512, activation="sigmoid")(inputs_RNN)
outputs_RNN = KL.Dense(512, activation="softmax")(x)
model_RNN = tf.keras.models.Model(inputs_RNN, outputs_RNN)
model_RNN.summary()
model_RNN.compile(optimizer = "adam", loss="sparse_categorical_crossentropy", metrics =["acc"])
model_RNN.fit(trainpose,labels_train, epochs=epochcount)#problem
test_loss, test_acc = model_RNN.evaluate(testpose, labels_test)
print("RRN Architecture's Acc%: {} for {} epochs".format(test_acc,epochcount))

'''
