import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import cvxpy as cp
import itertools

from interval import interval, inf

# from safe_model import SafeModel

MEAN = np.array([0.0, 0.0, 0.0, 20.0])
RANGES = np.array([16000.0, 200.0, 200.0, 40.0])


def generate_data(NOISE_STD=2, M=0.5, B=5, xmin=5, xmax=55, n=30):
    x = np.linspace(xmin, xmax, n)
    y_func = lambda x: M * x + B
    y_noisy = lambda x: y_func(x) + np.random.normal(0, NOISE_STD, np.shape(x))
    # y = np.array([5, 20, 14, 32, 22, 38])
    y = y_noisy(x)
    return x, y


def propagate_interval(input_interval, model, graph=False, verbose=False):
    # TODO check relu handling for multiple intervals
    # TODO change inputs to lists?

    # Complain if input_interval is not a list
    if type(input_interval) is not list:
        if verbose:
            print("Warning! Input interval was not a list")

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
                num_intervals = len(current_interval)
                if num_intervals == 1:
                    current_interval = [
                        (current_interval[0] - float(norm_mean)) / float(norm_std)
                    ]
                else:
                    assert len(norm_std) == len(current_interval)
                    intervals = [0] * num_intervals
                    for i in range(num_intervals):
                        if current_interval[i] is not None:
                            intervals[i] += (
                                current_interval[i] - norm_mean[i]
                            ) / norm_std[i]
                    current_interval = intervals
            else:
                current_interval = [
                    (current_interval - float(norm_mean)) / float(norm_std)
                ]
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
                            # print(
                            #     f"current interval is {current_interval[i]} with type {type(current_interval[i])}"
                            # )
                            # print(f"this weight is {weight[i, j]}")
                            intervals[j] += current_interval[i] * float(weight[i, j])
                for j in range(num_intervals):
                    intervals[j] += float(bias[j])
                current_interval = intervals
            if config["activation"] == "relu":
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
    if type(penultimate_interval) is not list:
        penultimate_interval = [penultimate_interval]
    return current_interval, penultimate_interval


def check_intervals(output_interval, goal_interval):
    assert type(output_interval) == type(goal_interval)
    if type(output_interval) is list:
        assert len(output_interval) == len(goal_interval)
        for i in range(len(output_interval)):
            if (
                goal_interval[i] is not None
                and output_interval[i] not in goal_interval[i]
            ):
                return False
        return True
    else:
        return output_interval in goal_interval


def check_max_score(output_interval, max_idx):
    assert type(output_interval) is list, "Output interval was not list"
    max_interval = output_interval[max_idx]
    # Check if all other intervals have sups below max_interval's inf
    # iterate: 0 -> max_idx-1
    for advisory_interval in output_interval[:max_idx]:
        if advisory_interval[0].sup > max_interval[0].inf:
            return False

    # iterate: max_idx + 1 -> end
    for advisory_interval in output_interval[max_idx + 1 :]:
        if advisory_interval[0].sup > max_interval[0].inf:
            return False

    return True


def normalize_point(x: np.array):
    if type(x) is list:
        x = np.array(x)
    return (x - MEAN) / RANGES


def denormalize_point(p: np.array):
    if type(p) is list:
        p = np.array(p)
    return p * RANGES + MEAN


def normalize_interval(ivls: list):
    if type(ivls) is list:
        assert len(ivls) == 4
    new_ivls = []
    for i in range(4):
        new_ivls.append((ivls[i] - MEAN[i]) / RANGES[i])
    return new_ivls
