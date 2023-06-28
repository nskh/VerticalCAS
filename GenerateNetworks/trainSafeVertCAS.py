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


######## OPTIONS #########
ver = 4  # Neural network version
hu = 45  # Number of hidden units in each hidden layer in network
saveEvery = 5  # Epoch frequency of saving
totalEpochs = 50  # Total number of training epochs
EPOCH_TO_PROJECT = 5
trainingDataFiles = (
    "../TrainingData/VertCAS_TrainingData_v2_%02d.h5"  # File format for training data
)
nnetFiles = (
    "../networks/VertCAS_pra%02d_v%d_45HU_%03d.nnet"  # File format for .nnet files
)
##########################


def propagate_interval(input_interval, model, graph=False):
    # TODO check relu handling for multiple intervals
    # TODO change inputs to lists?
    num_layers = len(model.layers)
    current_interval = input_interval
    for layer_idx, layer in enumerate(model.layers):
        # print(current_interval)
        if layer_idx == num_layers - 1:
            penultimate_interval = current_interval
        config = layer.get_config()
        if "normalization" in config["name"]:
            # print(f"on normalization layer {layer_idx}")
            if graph:
                norm_mean, norm_var, _ = layer.weights
            else:
                norm_mean, norm_var, _ = layer.get_weights()
            norm_std = np.sqrt(norm_var)
            if type(current_interval) == list:
                assert len(norm_std) == len(current_interval)
                num_intervals = len(current_interval)
                if num_intervals == 1:
                    current_interval = [
                        (current_interval[0] - float(norm_mean)) / float(norm_std)
                    ]
                else:
                    intervals = [0] * num_intervals
                    for i in range(num_intervals):
                        if current_interval[i] is not None:
                            intervals[i] += (
                                current_interval[i] - norm_mean[i]
                            ) / norm_std[i]
                    current_interval = intervals
        elif "dense" in config["name"]:
            if graph:
                weight, bias = layer.weights
            else:
                weight, bias = layer.get_weights()
            num_combinations = weight.shape[0]
            num_intervals = weight.shape[1]
            # print(
            #     f"on dense layer {layer_idx} of dim ({num_combinations}x{num_intervals})"
            # )
            if num_combinations == 1 and num_intervals == 1:
                if type(current_interval) == list:
                    assert (
                        len(current_interval) == 1
                    ), f"Expected only one interval, got {len(current_interval)}"
                    current_interval = current_interval[0]
                current_interval = [current_interval * float(weight) + float(bias)]
            # elif num_combinations == 1 and num_intervals > 1:
            #     # Make multiple intervals
            #     intervals = []
            #     assert type(current_interval) is list
            #     assert len(current_interval) == 1
            #     for i in range(num_intervals):
            #         intervals.append(current_interval[0] * weight[0, i] + bias[i])
            #     current_interval = intervals
            #     assert (
            #         type(current_interval) is list
            #     ), "Current interval was not type list"
            #     assert (
            #         len(current_interval) == num_intervals
            #     ), "Length of intervals was wrong"
            # elif num_combinations > 1 and num_intervals == 1:
            #     assert (
            #         type(current_interval) == list
            #     ), "Current interval was not type list"
            #     # start at 0
            #     interval = 0
            #     for i in range(num_combinations):
            #         interval += current_interval[i] * weight[i, 0]
            #     interval += bias
            #     current_interval = interval
            else:
                intervals = [0] * num_intervals
                for i in range(num_combinations):
                    # print(f"comb {i}")
                    for j in range(num_intervals):
                        # print(f"interval {j}")
                        if current_interval[i] is not None:
                            intervals[j] += current_interval[i] * float(weight[i, j])
                for j in range(num_intervals):
                    intervals[j] += float(bias[j])
                current_interval = intervals
            if config["activation"] == "relu":
                # TODO test this handling inside loop!!
                # can't do a for-in here since that does a copy
                for interval_idx in range(len(current_interval)):
                    current_interval[interval_idx] &= interval[0, inf]
                    if current_interval[interval_idx] == interval():
                        current_interval[interval_idx] = interval[0, 0]
            elif config["activation"] == "linear":
                # Do nothing, just pass interval through
                pass
            else:
                raise NotImplementedError(
                    f"Activation type {config['activation']} is not handled"
                )
        elif "input" in config["name"]:
            # Do nothing, just pass interval through
            pass
        else:
            raise NotImplementedError(f"Layer type {config['name']} is not handled")
    return current_interval, penultimate_interval


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
    print(f"x_train shape: {X_train.shape}")
    print(f"min inputs: {min_inputs}")
    print(f"max inputs: {max_inputs}")

    N, numOut = Q.shape
    print(f"Setting up model with {numOut} outputs")

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

    # TODO add batch size handling
    for epoch in range(totalEpochs):
        print(f"on epoch {epoch}")
        with tf.GradientTape() as tape:
            y_pred = model(X_train, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = asymMSE(Q, y_pred)

        # Compute gradients
        trainable_vars = model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        opt.apply_gradients(zip(gradients, trainable_vars))

        # Parameters:
        # - h (ft): Altitude of intruder relative to ownship, [-8000, 8000]
        # - vO (ft/s): ownship vertical climb rate, [-100, 100]
        # - vI (ft/s): intruder vertical climb rate, [-100, 100]
        # - τ (sec): time to loss of horizontal separation
        output_interval, penultimate_interval = propagate_interval(
            [
                interval[7880, 7900],
                interval[95, 96],
                interval[5, 6],
                interval[38, 40],
            ],
            model,
            graph=False,
        )
        if epoch % 10 == 0:
            print(output_interval)
        # if type(output_interval) is list:
        #     if len(output_interval) == 1:
        #         output_interval = output_interval[0]
        #     else:
        #         raise NotImplementedError("Output interval was interval of length > 1")
        # if output_interval not in desired_interval:
        #     print(f"safe region test FAILED, interval was {output_interval}")
        #     print(model.layers[-1].weights)
        # if epoch % EPOCH_TO_PROJECT == 0:
        #     print(f"\nProjecting weights at epoch {epoch}.")
        #     weights = model.layers[-1].weights
        #     print(
        #         f"Old weights: {np.squeeze(np.array([weight.numpy() for weight in weights]))}"
        #     )
        #     projected_weights = project_weights(
        #         desired_interval,
        #         penultimate_interval,
        #         np.squeeze(np.array(weights)),
        #     )
        #     print(
        #         f"Projected weights: {projected_weights} yield new interval: "
        #         f"{penultimate_interval * projected_weights[0] + projected_weights[1]}"
        #     )
        #     proj_weight, proj_bias = projected_weights
        #     model.layers[-1].set_weights(
        #         [np.array([[proj_weight]]), np.array([proj_bias])]
        #     )
        #     # NOTE: assume positive weights
        #     # TODO: handle both signs of weights

        #     # print(optimizer.get_weights())
        #     # optimizer.set_weights(last_safe_weights)
        # else:
        #     print(f"safe region test passed, interval was {output_interval}")

        # Update metrics (includes the metric that tracks the loss)
        # model.compiled_metrics.update_state(Q, y_pred)
        # Return a dict mapping metric names to current value

    # # Train and write nnet files
    # epoch = saveEvery
    # while epoch <= totalEpochs:
    #     model.fit(X_train, Q, epochs=saveEvery, batch_size=2**8, shuffle=True)
    #     saveFile = nnetFiles % (pra, ver, epoch)
    #     saveNNet(model, saveFile, means, ranges, min_inputs, max_inputs)
    #     epoch += saveEvery
