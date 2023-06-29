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

from interval import interval, inf

from safe_train import propagate_interval

######## OPTIONS #########
ver = 4  # Neural network version
hu = 45  # Number of hidden units in each hidden layer in network
saveEvery = 3  # Epoch frequency of saving
totalEpochs = 20  # Total number of training epochs
BATCH_SIZE = 2**8
EPOCH_TO_PROJECT = 5
trainingDataFiles = (
    "../TrainingData/VertCAS_TrainingData_v2_%02d.h5"  # File format for training data
)
nnetFiles = (
    "../networks/SafeVertCAS_pra%02d_v%d_45HU_%03d.nnet"  # File format for .nnet files
)
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

    # for epoch in range(totalEpochs):
    #     # if epoch % 5 == 0:
    #     print(f"on epoch {epoch}")

    #     rng = np.random.default_rng()

    #     train_indices = np.arange(X_train.shape[0])

    #     rng.shuffle(train_indices)  # in-place

    #     x_shuffled = X_train[train_indices, :]
    #     y_shuffled = Q[train_indices, :]

    #     x_batched = np.split(
    #         x_shuffled, np.arange(BATCH_SIZE, len(x_shuffled), BATCH_SIZE)
    #     )
    #     y_batched = np.split(
    #         y_shuffled, np.arange(BATCH_SIZE, len(y_shuffled), BATCH_SIZE)
    #     )

    #     dataset_batched = list(zip(x_batched, y_batched))
    #     for step, (x_batch_train, y_batch_train) in enumerate(dataset_batched):
    #         if step % int(num_batches / 100) == 0:
    #             print(
    #                 f"{np.round(step / num_batches * 100, 2)}% through this epoch\r",
    #                 end="",
    #             )
    #         with tf.GradientTape() as tape:
    #             y_pred = model(x_batch_train, training=True)  # Forward pass
    #             # Compute the loss value
    #             # (the loss function is configured in `compile()`)
    #             loss = asymMSE(y_batch_train, y_pred)

    #         # Compute gradients
    #         trainable_vars = model.trainable_variables
    #         gradients = tape.gradient(loss, trainable_vars)
    #         # Update weights
    #         opt.apply_gradients(zip(gradients, trainable_vars))

    #     # Parameters:
    #     # - h (ft): Altitude of intruder relative to ownship, [-8000, 8000]
    #     # - vO (ft/s): ownship vertical climb rate, [-100, 100]
    #     # - vI (ft/s): intruder vertical climb rate, [-100, 100]
    #     # - τ (sec): time to loss of horizontal separation
    #     output_interval, penultimate_interval = propagate_interval(
    #         [
    #             interval[7880, 7900],
    #             interval[95, 96],
    #             interval[5, 6],
    #             interval[38, 40],
    #         ],
    #         model,
    #         graph=False,
    #     )
    #     if epoch % 10 == 0:
    #         print(output_interval)
    #     if type(output_interval) is list:
    #         if len(output_interval) == 1:
    #             output_interval = output_interval[0]
    #         else:
    #             raise NotImplementedError("Output interval was interval of length > 1")
    #     if output_interval not in desired_interval:
    #         print(f"safe region test FAILED, interval was {output_interval}")
    #         print(model.layers[-1].weights)
    #     if epoch % EPOCH_TO_PROJECT == 0:
    #         print(f"\nProjecting weights at epoch {epoch}.")
    #         weights = model.layers[-1].weights
    #         print(
    #             f"Old weights: {np.squeeze(np.array([weight.numpy() for weight in weights]))}"
    #         )
    #         projected_weights = project_weights(
    #             desired_interval,
    #             penultimate_interval,
    #             np.squeeze(np.array(weights)),
    #         )
    #         print(
    #             f"Projected weights: {projected_weights} yield new interval: "
    #             f"{penultimate_interval * projected_weights[0] + projected_weights[1]}"
    #         )
    #         proj_weight, proj_bias = projected_weights
    #         model.layers[-1].set_weights(
    #             [np.array([[proj_weight]]), np.array([proj_bias])]
    #         )
    #         # NOTE: assume positive weights
    #         # TODO: handle both signs of weights

    #         # print(optimizer.get_weights())
    #         # optimizer.set_weights(last_safe_weights)
    #     else:
    #         print(f"safe region test passed, interval was {output_interval}")

    #     Update metrics (includes the metric that tracks the loss)
    #     model.compiled_metrics.update_state(Q, y_pred)
    #     Return a dict mapping metric names to current value

    # # Train and write nnet files
    epoch = saveEvery
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
