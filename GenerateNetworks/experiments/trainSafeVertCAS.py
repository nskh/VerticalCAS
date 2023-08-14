import sys
import configparser
import math
import os
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
from GenerateNetworks.utils.utils import *

config = load_config()

######## OPTIONS #########
ver = 4  # Neural network version
hu = 45  # Number of hidden units in each hidden layer in network
saveEvery = 3  # Epoch frequency of saving
totalEpochs = 20  # Total number of training epochs
BATCH_SIZE = 2**8
EPOCH_TO_PROJECT = 5
lossFactor = 40.0
learningRate = 0.0003

trainingDataFiles = (
    os.path.join(config['Paths']["training_data_dir"], "VertCAS_TrainingData_v2_%02d.h5")
)
nnetFiles = os.path.join(config["Paths"]["networks_dir"], "ProjectionVertCAS_pra%02d_v%d_45HU_%03d.nnet")

advisories = {
    "COC": 0,
    "DNC": 1,
    "DND": 2,
    "DES1500": 3,
    "CL1500": 4,
    "SDES1500": 5,
    "SCL1500": 6,
    "SDES2500": 7,
    "SCL2500": 8,
}
##########################


# The previous RA should be given as a command line input
if len(sys.argv) > 1:
    pra = int(sys.argv[1])

    X_train, Q, means, ranges, min_inputs, max_inputs = load_training_data(pra, trainingDataFiles, ver)

    N, numOut = Q.shape
    print(f"Setting up model with {numOut} outputs and {N} training examples")
    num_batches = N / BATCH_SIZE

    opt = Nadam(lr=learningRate)
    model = create_model(numOut, hu, learningRate, lossFactor, opt)

    epoch = saveEvery
    # TODO epoch numbering is wonky here
    while epoch <= totalEpochs:
        model.fit(X_train, Q, epochs=saveEvery, batch_size=2**8, shuffle=True)
        saveFile = nnetFiles % (pra, ver, epoch)
        saveNNet(model, saveFile, means, ranges, min_inputs, max_inputs)
        epoch += saveEvery
        output_interval, penultimate_interval = propagate_interval(
            [
                interval[7880, 7900],
                interval[95, 96],
                interval[-96, -95],
                interval[19, 20],
            ],
            model,
            graph=False,
        )
        print(output_interval)
