import numpy as np
import cvxpy as cp


def generate_constraints(goal_interval, input_intervals, x, theta, verbose=False):
    lowers = []
    uppers = []
    for i, ivl in enumerate(input_intervals):
        if theta[i] < 0:
            # if weight is negative
            # swap order of interval bounds used in constraints
            lowers.append(ivl[0][1])
            uppers.append(ivl[0][0])
        else:
            lowers.append(ivl[0][0])
            uppers.append(ivl[0][1])

    interval_combinations = [lowers, uppers]
    # no bias projections, only weights
    constraint_vectors = [np.hstack([elem]) for elem in interval_combinations]
    # constraint_vectors = [np.hstack([elem, 1]) for elem in interval_combinations]
    constraints = []
    if verbose:
        print(f"Generated {len(constraint_vectors)} constraint orderings")
    for constraint_vector in constraint_vectors:
        constraints.append(constraint_vector @ x >= goal_interval[0][0])
        constraints.append(constraint_vector @ x <= goal_interval[0][1])
        if verbose:
            print(f"{constraint_vector} @ x >= {goal_interval[0][0]}")
            print(f"{constraint_vector} @ x <= {goal_interval[0][1]}")

    for i in range(len(theta) - 1):
        constraint_row = np.zeros(theta.shape)
        np.put(constraint_row, i, 1)
        if theta[i] >= 0:
            # enforce weight stays positive during optimization
            constraints.append(constraint_row @ x >= 0)
        else:
            # enforce weight stays negative optimization
            constraints.append(constraint_row @ x <= 0)

    return constraints


def project_weights(goal_interval, input_intervals, theta, verbose=False):
    x = cp.Variable(theta.shape)
    constraints = generate_constraints(
        goal_interval, input_intervals, x, theta, verbose
    )
    obj = cp.Minimize(cp.norm(x - theta) ** 2)
    prob = cp.Problem(obj, constraints)
    prob.solve()  # Returns the optimal value.
    return x.value


def project_weights_vector(goal_interval, input_intervals, theta, verbose=False):
    shift_lower = np.array([0, goal_interval[0].inf])
    print(f"input interval: {input_intervals}")
    direction_lower = np.array([1, -input_intervals[0].inf])
    project_lower = (
        (direction_lower @ (theta - shift_lower))
        / (direction_lower @ direction_lower)
        * direction_lower
    )
    param_lower = project_lower + shift_lower

    shift_upper = np.array([0, goal_interval[0].sup])
    direction_upper = np.array([1, -input_intervals[0].sup])
    project_upper = (
        (direction_upper @ (theta - shift_upper))
        / (direction_upper @ direction_upper)
        * direction_upper
    )
    param_upper = project_upper + shift_upper

    return min(
        [param_upper, param_lower], key=lambda param: np.linalg.norm(theta - param)
    )
