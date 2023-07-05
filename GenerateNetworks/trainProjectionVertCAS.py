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
from writeNNet import saveNNet

from interval import interval, inf

from safe_train import propagate_interval, check_intervals, project_weights

######## OPTIONS #########
ver = 4  # Neural network version
hu = 45  # Number of hidden units in each hidden layer in network
saveEvery = 1  # Epoch frequency of saving
totalEpochs = 20  # Total number of training epochs
BATCH_SIZE = 2**8
EPOCH_TO_PROJECT = 2
trainingDataFiles = (
    "../TrainingData/VertCAS_TrainingData_v2_%02d.h5"  # File format for training data
)
nnetFiles = "../networks/ProjectionVertCAS_pra%02d_v%d_45HU_%03d.nnet"  # File format for .nnet files
COC_INTERVAL = [
    interval[7880, 7900],
    interval[95, 96],
    interval[5, 6],
    interval[38, 40],
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
    print("Loading Data for VertCAS, pra %02d, Network Version %d" % (pra, ver))
    f = h5py.File(trainingDataFiles % pra, "r")
    X_train = np.array(f["X"])
    Q = np.array(f["y"])
    means = np.array(f["means"])
    ranges = np.array(f["ranges"])
    min_inputs = np.array(f["min_inputs"])
    max_inputs = np.array(f["max_inputs"])
    print(f"min inputs: {min_inputs}")
    print(f"max inputs: {max_inputs}")

    N, numOut = Q.shape
    print(f"Setting up model with {numOut} outputs and {N} training examples")
    num_batches = N / BATCH_SIZE

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

    epoch_losses = []
    epoch_accuracies = []
    weights_before_projection = []
    weights_after_projection = []
    for epoch in range(totalEpochs):
        # if epoch % 5 == 0:
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
                loss = asymMSE(y_batch_train, y_pred)
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
            COC_INTERVAL,
            model,
            graph=False,
        )
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
                    weights_to_project = np.hstack([weights_np[:, idx], biases_np[idx]])
                    proj = project_weights(
                        desired_interval[idx], penultimate_interval, weights_to_project
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

        with open("projection_acas.pickle", "wb") as f:
            data = {
                "accuracies": epoch_accuracies,
                "losses": epoch_losses,
                "weights_before_projection": weights_before_projection,
                "weights_after_projection": weights_after_projection,
            }
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
