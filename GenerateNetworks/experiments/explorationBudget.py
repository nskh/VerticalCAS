import numpy as np
import math
import tensorflow as tf
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
import h5py
from keras.optimizers import Adamax, Nadam
import sys
import pickle

from interval import interval, inf

from GenerateNetworks.utils.safe_train import propagate_interval, check_intervals
from GenerateNetworks.utils.projection_utils import project_weights
from GenerateNetworks.writeNNet import saveNNet
from GenerateNetworks.utils.utils import *

# NOTE marabou must be added to the python path
# export PYTHONPATH=$PYTHONPATH:~/VerticalCAS/GenerateNetworks/Marabou
# TODO find a way to nicely do this automatically

from maraboupy import Marabou, MarabouCore

config = load_config()

######## OPTIONS #########
ver = 4  # Neural network version
hu = 45  # Number of hidden units in each hidden layer in network
saveEvery = 1  # Epoch frequency of saving
totalEpochs = 20  # Total number of training epochs
BATCH_SIZE = 2**8
EPOCH_TO_PROJECT = 2
lossFactor = 40.0
learningRate = 0.0003

trainingDataFiles = (
    os.path.join(config['Paths']["training_data_dir"], "VertCAS_TrainingData_v2_%02d.h5")
)
nnetFiles = os.path.join(config["Paths"]["networks_dir"], "ProjectionVertCAS_pra%02d_v%d_45HU_%03d.nnet")

COC_INTERVAL = [
    interval[400, 500],
    interval[50, 51],
    interval[-51, -50],
    interval[20, 21],
]
# COC high, SDES2500 low
desired_interval = [
    interval[7000, 15000],  # COC
    None,  # DNC
    None,  # DND
    None,  # DES1500
    None,  # CL1500
    None,  # SDES1500
    None,  # SCL1500
    interval[-2000, 6000],  # SDES2500
    None,  # SCL2500
]
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

    opt = Nadam(learning_rate=learningRate)
    model = create_model(numOut, hu, learningRate, lossFactor, opt)

    last_safe_weights = None
    last_safe_epoch = 0
    num_unsafe_epochs = 0

    epoch_losses = []
    epoch_accuracies = []
    weights_before_projection = []
    for epoch in range(totalEpochs):
        print(f"on epoch {epoch}")

        rng = np.random.default_rng()

        train_indices = np.arange(X_train.shape[0])

        rng.shuffle(train_indices)  # in-place

        x_shuffled = X_train[train_indices, :]
        y_shuffled = Q[train_indices, :]

        x_batched = np.split(
            x_shuffled, np.arange(BATCH_SIZE, len(x_shuffled), BATCH_SIZE)
        )
        y_batched = np.split(
            y_shuffled, np.arange(BATCH_SIZE, len(y_shuffled), BATCH_SIZE)
        )

        dataset_batched = list(zip(x_batched, y_batched))
        batch_losses = []
        batch_accuracy_list = []
        epoch_accuracy = keras.metrics.CategoricalAccuracy()
        for step, (x_batch_train, y_batch_train) in enumerate(dataset_batched):
            with tf.GradientTape() as tape:
                y_pred = model(x_batch_train, training=True)  # Forward pass
                loss = asymMSE(y_batch_train, y_pred, numOut, lossFactor)
                epoch_accuracy.update_state(y_batch_train, y_pred)

                # accumulate data
                batch_losses.append(loss.numpy())
                batch_accuracy_list.append(epoch_accuracy.result())
            if step % int(num_batches / 500) == 0:
                print(
                    f"{np.round(step / num_batches * 100, 1)}% through this epoch with loss",
                    f"{np.round(loss.numpy(), 5)} and accuracy {np.round(epoch_accuracy.result(), 5)}\r",
                    end="",
                )
            # Compute gradients
            trainable_vars = model.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            # Update weights
            opt.apply_gradients(zip(gradients, trainable_vars))

        epoch_accuracies.append(batch_accuracy_list)
        epoch_losses.append(batch_losses)

        weights_before_projection.append([w.numpy() for w in model.layers[-1].weights])

        if epoch == 0:
            last_safe_weights = model.get_weights()

        # TODO here downward for Marabou integration

        # Parameters:
        # - h (ft): Altitude of intruder relative to ownship, [-8000, 8000]
        # - vO (ft/s): ownship vertical climb rate, [-100, 100]
        # - vI (ft/s): intruder vertical climb rate, [-100, 100]
        # - τ (sec): time to loss of horizontal separation
        output_interval, penultimate_interval = propagate_interval(
            COC_INTERVAL,
            model,
            graph=False,
        )

        print("With Marabou:\n")
        tf.saved_model.save(model, "tmp")
        network = Marabou.read_tf("tmp", modelType="savedModel_v2")

        inputVars = network.inputVars[0][0]
        outputVars = network.outputVars[0][0]

        print("input constraints")
        for i, in_int in enumerate(COC_INTERVAL):
            print(inputVars[i], ">", in_int[0].inf)
            network.setLowerBound(inputVars[i], in_int[0].inf)
            print(inputVars[i], "<", in_int[0].sup)
            network.setUpperBound(inputVars[i], in_int[0].sup)

        print("output constraints")
        for i, des_int in enumerate(desired_interval):
            if des_int is None:
                continue
            print(outputVars[i], ">", des_int[0].inf)
            print(outputVars[i], "<", des_int[0].sup)

            ineq1 = MarabouCore.Equation(MarabouCore.Equation.LE)
            ineq1.addAddend(outputVars[i], 1)
            ineq1.setScalar(des_int[0].inf)

            ineq2 = MarabouCore.Equation(MarabouCore.Equation.GE)
            ineq2.addAddend(outputVars[i], 1)
            ineq2.setScalar(des_int[0].sup)
            disjunction = [[ineq1], [ineq2]]
            network.addDisjunctionConstraint(disjunction)

        # Check relative ordering of outputs?
        # print("Add max constraint")
        # print(f"{outputVars[1]} should be max among outputVars")
        # the_max_var_idx = 1
        # for i, var in enumerate(outputVars):
        #     if i == the_max_var_idx:
        #         continue
        #     print(f"{outputVars[the_max_var_idx]} - {outputVars[i]} < 0")
        #     network.addInequality([outputVars[the_max_var_idx], outputVars[i]], [1, -1], 0)

        _, vals, stats = network.solve("marabou.log")
        if vals is None:
            print("UNSAT. So safe region test passed.")
            last_safe_weights = model.get_weights()
            last_safe_epoch = epoch
            num_unsafe_epochs = 0
        else:
            print(f"safe region test FAILED, counterexample {vals}")
            num_unsafe_epochs += 1

        if num_unsafe_epochs == 10:
            print("Exploration budget exhausted.")
            print("Restarting training from last safe epoch.")
            model.set_weights(last_safe_weights)
            num_unsafe_epochs = 0
        else:
            model.compiled_metrics.update_state(y, y_pred)

        with open("exploration_budget_acas.pickle", "wb") as f:
            data = {
                "accuracies": epoch_accuracies,
                "losses": epoch_losses,
                "weights_before_projection": weights_before_projection,
            }
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
