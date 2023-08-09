import numpy as np
import math
import tensorflow as tf
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
import h5py
from keras.optimizers import Adamax, Nadam
import sys
from writeNNet import saveNNet
import maraboupy
######## OPTIONS #########
ver = 4  # Neural network version
hu = 45  # Number of hidden units in each hidden layer in network
saveEvery = 5  # Epoch frequency of saving
totalEpochs = 50  # Total number of training epochs
trainingDataFiles = (
    "../TrainingData/VertCAS_TrainingData_v2_%02d.h5"  # File format for training data
)
nnetFiles = (
    "../networks/VertCAS_pra%02d_v%d_45HU_%03d.nnet"  # File format for .nnet files
)
##########################

# The previous RA should be given as a command line input
if len(sys.argv) > 1:
    pra = int(sys.argv[1])
    print("Loading Data for VertCAS, pra %02d, Network Version %d" % (pra, ver))
    f = h5py.File(trainingDataFiles % pra, "r")
    X_train = np.array(f["X"])
    Q = np.array(f["y"])
    means = np.array(f["means"])
    ranges = np.array(f["ranges"])
    min_inputs = np.array(f["min_inputs"])
    max_inputs = np.array(f["max_inputs"])

    N, numOut = Q.shape
    print("Setting up Model")

    # Asymmetric loss function
    lossFactor = 40.0

    # NOTE(nskh): from HorizontalCAS which was updated to use TF
    def asymMSE(y_true, y_pred):
        d = y_true - y_pred
        maxes = tf.argmax(y_true, axis=1)
        maxes_onehot = tf.one_hot(maxes, numOut)
        others_onehot = maxes_onehot - 1
        d_opt = d * maxes_onehot
        d_sub = d * others_onehot
        a = lossFactor * (numOut - 1) * (tf.square(d_opt) + tf.abs(d_opt))
        b = tf.square(d_opt)
        c = lossFactor * (tf.square(d_sub) + tf.abs(d_sub))
        d = tf.square(d_sub)
        loss = tf.where(d_sub > 0, c, d) + tf.where(d_opt > 0, a, b)
        return tf.reduce_mean(loss)

    # Define model architecture
    model = Sequential()
    # model.add(Dense(hu, init='uniform', activation='relu', input_dim=4))
    # model.add(Dense(hu, init='uniform', activation='relu'))
    # model.add(Dense(hu, init='uniform', activation='relu'))
    # model.add(Dense(hu, init='uniform', activation='relu'))
    # model.add(Dense(hu, init='uniform', activation='relu'))
    # model.add(Dense(hu, init='uniform', activation='relu'))
    model.add(Dense(hu, activation="relu", input_dim=4))
    model.add(Dense(hu, activation="relu"))
    model.add(Dense(hu, activation="relu"))
    model.add(Dense(hu, activation="relu"))
    model.add(Dense(hu, activation="relu"))
    model.add(Dense(hu, activation="relu"))

    # model.add(Dense(numOut, init="uniform"))
    model.add(Dense(numOut))
    opt = Nadam(learning_rate=0.0003)
    model.compile(loss=asymMSE, optimizer=opt, metrics=["accuracy"])

    # Train and write nnet files
    epoch = saveEvery
    while epoch <= totalEpochs:
        model.fit(X_train, Q, epochs=saveEvery, batch_size=2**8, shuffle=True)
        saveFile = nnetFiles % (pra, ver, epoch)
        saveNNet(model, saveFile, means, ranges, min_inputs, max_inputs)
        epoch += saveEvery
