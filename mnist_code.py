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
digepoch_acc =  [[98.2,97.0,92.6,90.01],[70.1,79.1,85.6,87.1,89.3],[91.2,95.5,97.2,96.1]]

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
digotmi_acc = [[],[89.3,85.39,82.39,71.39],[97.2,95.1,94.9]]

#Mode 3: RNN
############

inputs_RNN = KL.Input(shape=shapedim)
x = KL.SimpleRNN(512, activation="sigmoid")(inputs_RNN)
poseotmi_acc = [[56.86,29.90,14.754,51.47],[1.47,1.47,14.22,30.39],[12.25,5.39,3.43,11.76]]
outputs_RNN = KL.Dense(512, activation="softmax")(x)
model_RNN = tf.keras.models.Model(inputs_RNN, outputs_RNN)
model_RNN.summary()
model_RNN.compile(optimizer = "adam", loss="sparse_categorical_crossentropy", metrics =["acc"])
model_RNN.fit(trainpose,labels_train, epochs=epochcount)#problem
test_loss, test_acc = model_RNN.evaluate(testpose, labels_test)


for x in typelist:
    print(x,"\n")
    for y in poseepoch_acc[typelist.index(x)]:      
        print("EPOCH {} ACC%  -- {}".format(poseepochlist[poseepoch_acc[typelist.index(x)].index(y)],y))
    print("\n")
    for y in poseotmi_acc[typelist.index(x)]:       
        print("OPTIMIZER {} ACC%  -- {}".format(poseotmilist[poseotmi_acc[typelist.index(x)].index(y)],y))
    print("\n------------\n")


for x in typelist:
    print(x,"\n")
    for y in digepoch_acc[typelist.index(x)]:       
        print("EPOCH {} ACC%  -- {}".format(digepochlist[digepoch_acc[typelist.index(x)].index(y)],y))
    print("\n")
    for y in digotmi_acc[typelist.index(x)]:        
        print("OPTIMIZER {} ACC%  -- {}".format(digotmilist[digotmi_acc[typelist.index(x)].index(y)],y))
    print("\n------------\n")

########################################################
########################################################

# ----------------- MNIST  ----------------- 
########################################################
########################################################


from keras.datasets import mnist
epochlist = [2,5,10,15,20]
otmilist = ["ADAM","STOC_GRAD","ADAGRAD","RMSPROP"]
typelist = ["CNN","FF","RRN"]




#Mode 1: CNN 
############


# load dataset
(trainX, trainy), (testX, testy) = mnist.load_data()
# summarize loaded dataset
print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))
# plot first few images
for i in range(9):
    # define subplot
    plt.subplot(330 + 1 + i)
    # plot raw pixel data
    plt.imshow(trainX[i], cmap=plt.get_cmap('gray'))
# show the figure
plt.show()

# record model performance on a validation dataset during training
history = model.fit(..., validation_data=(valX, valY))

# example of k-fold cv for a neural net
data = ...

# load dataset
(trainX, trainY), (testX, testY) = mnist.load_data()
# reshape dataset to have a single channel
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))
# load dataset
(trainX, trainY), (testX, testY) = mnist.load_data()
# reshape dataset to have a single channel
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))


# one hot encode target values
trainY = to_categorical(trainY)
testY = to_categorical(testY)

# one hot encode target values
trainY = to_categorical(trainY)
testY = to_categorical(testY)


# load train and test dataset
def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY

# define cnn model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# define cnn model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model



# run the test harness for evaluating a model
def run_test_harness():
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    # evaluate model
    scores, histories = evaluate_model(trainX, trainY)
    # learning curves
    summarize_diagnostics(histories)
    # summarize estimated performance


#Mode 2: Feed-forward NN 
############

import keras
import numpy as np
import pandas as pd 
data=pd.read_csv("../input/train.csv")
datat=pd.read_csv("../input/test.csv")

X_train=data.iloc[:,1:785]
y_train=data.iloc[:,0]
yt=keras.utils.to_categorical(y_train,10)
X_test=datat.iloc[:,0:785]
X_test

from keras import Sequential
from keras.layers import Dense
classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(32, activation='sigmoid', kernel_initializer='random_normal', input_dim=784))
#Output Layer
classifier.add(Dense(10, activation='softmax', kernel_initializer='random_normal'))
#Compiling the neural network
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
classifier.fit(X_train,yt, batch_size=10, epochs=20)


#Mode 3: RNN 
############

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

# hyperparameters
n_neurons = 128
learning_rate = 0.001
batch_size = 128
n_epochs = 10
# parameters
n_steps = 28 # 28 rows
n_inputs = 28 # 28 cols
n_outputs = 10 # 10 classes
# build a rnn model
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
output, state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
logits = tf.layers.dense(state, n_outputs)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
prediction = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

# input data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(“MNIST_data/”)
X_test = mnist.test.images # X_test shape: [num_test, 28*28]
X_test = X_test.reshape([-1, n_steps, n_inputs])
y_test = mnist.test.labels

# initialize the variables
init = tf.global_variables_initializer()
# train the model
with tf.Session() as sess:
    sess.run(init)
    n_batches = mnist.train.num_examples // batch_size
    for epoch in range(n_epochs):
        for batch in range(n_batches):
            X_train, y_train = mnist.train.next_batch(batch_size)
            X_train = X_train.reshape([-1, n_steps, n_inputs])
            sess.run(optimizer, feed_dict={X: X_train, y: y_train})
        loss_train, acc_train = sess.run(
            [loss, accuracy], feed_dict={X: X_train, y: y_train})
