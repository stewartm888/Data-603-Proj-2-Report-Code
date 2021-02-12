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


##CNN
###################
###################

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
digotmi_acc = [[],[89.3,85.39,82.39,71.39],
[97.2,95.1,94.9]]
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


### Feed Forward
#######################
from sklearn.datasets import fetch_openml
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
import time

x, y = fetch_openml('mnist_784', version=1, return_X_y=True)
x = (x/255).astype('float32')
digepoch_acc =  [[98.2,97.0,92.6,90.01],
[70.1,79.1,85.6,87.1,89.3],[91.2,95.5,97.2,96.1]]
y = to_categorical(y)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=42)

class DeepNeuralNetwork():
    def __init__(self, sizes, epochs=10, l_rate=0.001):
        self.sizes = sizes
        self.epochs = epochs
        self.l_rate = l_rate

        self.params = self.initialization()

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))

    def softmax(self, x, derivative=False):
        # Numerically stable with large exponentials
        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

    def initialization(self):
        # number of nodes in each layer
        input_layer=self.sizes[0]
        hidden_1=self.sizes[1]
        hidden_2=self.sizes[2]
        output_layer=self.sizes[3]

        params = {
            'W1':np.random.randn(hidden_1, input_layer) * np.sqrt(1. / hidden_1),
            'W2':np.random.randn(hidden_2, hidden_1) * np.sqrt(1. / hidden_2),
            'W3':np.random.randn(output_layer, hidden_2) * np.sqrt(1. / output_layer)
        }

        return params

    def forward_pass(self, x_train):
        params = self.params

        params['A0'] = x_train

        # input layer to hidden layer 1
        params['Z1'] = np.dot(params["W1"], params['A0'])
        params['A1'] = self.sigmoid(params['Z1'])

        # hidden layer 1 to hidden layer 2
        params['Z2'] = np.dot(params["W2"], params['A1'])
        params['A2'] = self.sigmoid(params['Z2'])

        # hidden layer 2 to output layer
        params['Z3'] = np.dot(params["W3"], params['A2'])
        params['A3'] = self.softmax(params['Z3'])

        return params['A3']

    def backward_pass(self, y_train, output):

        params = self.params
        change_w = {}

        # Calculate W3 update
        error = 2 * (output - y_train) / output.shape[0] * self.softmax(params['Z3'], derivative=True)
        change_w['W3'] = np.outer(error, params['A2'])

        # Calculate W2 update
        error = np.dot(params['W3'].T, error) * self.sigmoid(params['Z2'], derivative=True)
        change_w['W2'] = np.outer(error, params['A1'])

        # Calculate W1 update
        error = np.dot(params['W2'].T, error) * self.sigmoid(params['Z1'], derivative=True)
        change_w['W1'] = np.outer(error, params['A0'])

        return change_w

    def update_network_parameters(self, changes_to_w):

        for key, value in changes_to_w.items():
            self.params[key] -= self.l_rate * value

    def compute_accuracy(self, x_val, y_val):

        predictions = []

        for x, y in zip(x_val, y_val):
            output = self.forward_pass(x)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(y))
        
        return np.mean(predictions)

    def train(self, x_train, y_train, x_val, y_val):
        start_time = time.time()
        for iteration in range(self.epochs):
            for x,y in zip(x_train, y_train):
                output = self.forward_pass(x)
                changes_to_w = self.backward_pass(y, output)
                self.update_network_parameters(changes_to_w)
            
            accuracy = self.compute_accuracy(x_val, y_val)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%'.format(
                iteration+1, time.time() - start_time, accuracy * 100
            ))

dnn = DeepNeuralNetwork(sizes=[784, 128, 64, 10])
dnn.train(x_train, y_train, x_val, y_val)

import torch
from torchvision import datasets, transforms

transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transform))

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transform))

import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, epochs=10):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(784, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 10)

        self.epochs = epochs

    def forward_pass(self, x):
        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        x = self.linear3(x)
        x = torch.softmax(x, dim=0)
        return x
    
    def one_hot_encode(self, y):
        encoded = torch.zeros([10], dtype=torch.float64)
        encoded[y[0]] = 1.
        return encoded

    def train(self, train_loader, optimizer, criterion):
        start_time = time.time()
        loss = None

        for iteration in range(self.epochs):
            for x,y in train_loader:
                y = self.one_hot_encode(y)
                optimizer.zero_grad()
                output = self.forward_pass(torch.flatten(x))
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

            print('Epoch: {0}, Time Spent: {1:.2f}s, Loss: {2}'.format(
                iteration+1, time.time() - start_time, loss
            ))

model = Net()

optimizer = optim.SGD(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

model.train(train_loader, optimizer, criterion)


### RNN
##################

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

batch_size = 64
input_dim = 28

units = 64
output_size = 10  # labels are from 0 to 9

def build_model(allow_cudnn_kernel=True):
    if allow_cudnn_kernel:
        # The LSTM layer with default options uses CuDNN.
        lstm_layer = keras.layers.LSTM(units, input_shape=(None, input_dim))
    else:
        # Wrapping a LSTMCell in a RNN layer will not use CuDNN.
        lstm_layer = keras.layers.RNN(
            keras.layers.LSTMCell(units), input_shape=(None, input_dim)
        )
    model = keras.models.Sequential(
        [
            lstm_layer,
            keras.layers.BatchNormalization(),
            keras.layers.Dense(output_size),
        ]
    )
    return model

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
sample, sample_label = x_train[0], y_train[0]

model = build_model(allow_cudnn_kernel=True)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="sgd",
    metrics=["accuracy"],
)


model.fit(
    x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=10
)


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
