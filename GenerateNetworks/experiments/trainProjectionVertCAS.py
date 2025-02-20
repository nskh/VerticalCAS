import os
import sys
import configparser
import math
import numpy as np
import tensorflow as tf
import pickle
import h5py
from interval import interval, inf

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adamax, Nadam

from GenerateNetworks.writeNNet import saveNNet
from GenerateNetworks.utils.safe_train import (
    propagate_interval,
    check_intervals,
    check_max_score,
    normalize_interval,
)
from GenerateNetworks.utils.projection_utils import project_weights
from GenerateNetworks.utils.utils import *

config = load_config()

######## OPTIONS #########
ver = 4  # Neural network version
hu = 45  # Number of hidden units in each hidden layer in network
totalEpochs = 20  # Total number of training epochs
BATCH_SIZE = 2**8
EPOCH_TO_PROJECT = 2
lossFactor = 40.0
learningRate = 0.0003

trainingDataFiles = os.path.join(
    config["Paths"]["training_data_dir"], "VertCAS_TrainingData_v2_%02d.h5"
)
nnetFiles = os.path.join(
    config["Paths"]["networks_dir"], "ProjectionVertCAS_pra%02d_v%d_45HU_%03d.nnet"
)

COC_INTERVAL = [
    interval[-1000, -900],
    interval[50, 52],
    interval[-1, 1],
    interval[20, 22],
]
DES_INTERVAL = [
    interval[100, 110],
    interval[0, 0.5],
    interval[0, 0.5],
    interval[20, 21],
]

# COC high, SDES2500 low
desired_interval = [
    None,  # COC
    None,  # DNC
    None,  # DND
    interval[1, 10],  # DES1500
    None,  # CL1500
    None,  # SDES1500
    interval[-5, 0],  # SCL1500
    None,  # SDES2500
    None,  # SCL2500
]
# TODO make this dynamically set
INTERVAL_WIDTH = 5
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
relative = False
##########################

# The previous RA should be given as a command line input
if len(sys.argv) > 1:
    pra = int(sys.argv[1])
    X_train, Q, means, ranges, min_inputs, max_inputs = load_training_data(
        pra, trainingDataFiles, ver
    )

    N, numOut = Q.shape
    print(f"Setting up model with {numOut} outputs and {N} training examples")
    num_batches = N / BATCH_SIZE

    opt = Nadam(learning_rate=learningRate)
    model = create_model(numOut, hu, learningRate, lossFactor, opt)

    epoch_losses = []
    epoch_accuracies = []
    weights_before_projection = []
    weights_after_projection = []
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

        # Parameters:
        # - h (ft): Altitude of intruder relative to ownship, [-8000, 8000]
        # - vO (ft/s): ownship vertical climb rate, [-100, 100]
        # - vI (ft/s): intruder vertical climb rate, [-100, 100]
        # - τ (sec): time to loss of horizontal separation
        output_interval, penultimate_interval = propagate_interval(
            normalize_interval(
                DES_INTERVAL
            ),  # MAKE SURE to normalize interval for reasonable results
            model,
            graph=False,
        )
        if relative:
            if not check_max_score(output_interval, advisories["DES1500"]):
                print(f"safe region test FAILED, interval was {output_interval}")
                if epoch % EPOCH_TO_PROJECT == 0:
                    print(f"\nProjecting weights at epoch {epoch}.")
                    intervals_to_project = []
                    # go through all intervals, find max upper bound
                    max_upper_bound = None
                    for i, advisory_interval in enumerate(output_interval):
                        if i == advisories["DES1500"]:
                            continue
                        else:
                            if max_upper_bound is None:
                                max_upper_bound = advisory_interval[0].sup
                            else:
                                max_upper_bound = max(
                                    advisory_interval[0].sup, max_upper_bound
                                )

                    intervals_to_project = [advisories["DES1500"]]

                    weights_tf = model.layers[-1].weights
                    weights_np = weights_tf[0].numpy()
                    biases_np = weights_tf[1].numpy()

                    for idx in intervals_to_project:
                        weights_to_project = np.hstack(
                            [weights_np[:, idx], biases_np[idx]]
                        )
                        proj = project_weights(
                            interval[max_upper_bound, max_upper_bound + INTERVAL_WIDTH],
                            penultimate_interval,
                            weights_to_project,
                        )
                        weights_np[:, idx] = proj[:-1]
                        biases_np[idx] = proj[-1]

                    model.layers[-1].set_weights([weights_np, biases_np])
                    output_interval, _ = propagate_interval(
                        normalize_interval(DES_INTERVAL),
                        model,
                        graph=False,
                    )
                    print(f"After projecting, output interval is {output_interval}")
                    weights_after_projection.append(
                        [w.numpy() for w in model.layers[-1].weights]
                    )
        else:
            if not check_intervals(output_interval, desired_interval):
                print(f"safe region test FAILED, interval was {output_interval}")
                if epoch % EPOCH_TO_PROJECT == 0:
                    print(f"\nProjecting weights at epoch {epoch}.")
                    intervals_to_project = []
                    assert type(output_interval) == type(desired_interval)
                    if type(output_interval) is list:
                        assert len(output_interval) == len(desired_interval)
                        for i in range(len(output_interval)):
                            if (
                                desired_interval[i] is not None
                                and output_interval[i] not in desired_interval[i]
                            ):
                                intervals_to_project.append(i)
                    else:
                        intervals_to_project.append(0)

                    weights_tf = model.layers[-1].weights
                    weights_np = weights_tf[0].numpy()
                    biases_np = weights_tf[1].numpy()

                    for idx in intervals_to_project:
                        weights_to_project = np.hstack(
                            [weights_np[:, idx], biases_np[idx]]
                        )
                        proj = project_weights(
                            desired_interval[idx],
                            penultimate_interval,
                            weights_to_project,
                        )
                        weights_np[:, idx] = proj[:-1]
                        biases_np[idx] = proj[-1]

                    model.layers[-1].set_weights([weights_np, biases_np])
                    output_interval, _ = propagate_interval(
                        COC_INTERVAL,
                        model,
                        graph=False,
                    )
                    print(f"After projecting, output interval is {output_interval}")
                    weights_after_projection.append(
                        [w.numpy() for w in model.layers[-1].weights]
                    )

            else:
                print(f"safe region test passed, interval was {output_interval}")

        # Logging outputs
        with open("projection_acas_july6_coc.pickle", "wb") as f:
            data = {
                "accuracies": epoch_accuracies,
                "losses": epoch_losses,
                "weights_before_projection": weights_before_projection,
                "weights_after_projection": weights_after_projection,
            }
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
