import os
from functools import partial
import configparser
import h5py
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adamax, Nadam
from interval import interval, inf

from GenerateNetworks.writeNNet import saveNNet
from GenerateNetworks.utils.safe_train import propagate_interval

def load_config():
    config = configparser.ConfigParser()
    config.read(os.environ.get("CONFIG_INI_PATH"))
    return config

# NOTE(nskh): from HorizontalCAS which was updated to use TF
def asymMSE(y_true, y_pred, numOut, lossFactor):
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

def load_training_data(pra, trainingDataFiles, ver):
    print("Loading Data for VertCAS, pra %02d, Network Version %d" % (pra, ver))
    with h5py.File(trainingDataFiles % pra, "r") as f:
        X_train = np.array(f["X"])
        Q = np.array(f["y"])
        means = np.array(f["means"])
        ranges = np.array(f["ranges"])
        min_inputs = np.array(f["min_inputs"])
        max_inputs = np.array(f["max_inputs"])
        print(f"min inputs: {min_inputs}")
        print(f"max inputs: {max_inputs}")
    return X_train, Q, means, ranges, min_inputs, max_inputs

def create_model(numOut, hu, lr, lossFactor, opt):
    model = Sequential()
    model.add(Dense(hu, activation="relu", input_dim=4))
    for _ in range(5):
        model.add(Dense(hu, activation="relu"))
    model.add(Dense(numOut))
    model.compile(
        loss=partial(asymMSE, numOut = numOut, lossFactor = lossFactor),
        optimizer=opt,
        metrics=["accuracy"])
    return model
