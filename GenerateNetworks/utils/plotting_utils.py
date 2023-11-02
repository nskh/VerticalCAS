import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tensorflow as tf

# Constants
ra1 = (0.9, 0.9, 0.9)  # white
ra2 = (0.0, 1.0, 1.0)  # cyan
ra3 = (144.0 / 255.0, 238.0 / 255.0, 144.0 / 255.0)  # lightgreen
ra4 = (30.0 / 255.0, 144.0 / 255.0, 1.0)  # dodgerblue
ra5 = (0.0, 1.0, 0.0)  # lime
ra6 = (0.0, 0.0, 1.0)  # blue
ra7 = (34.0 / 255.0, 139.0 / 255.0, 34.0 / 255.0)  # forestgreen
ra8 = (0.0, 0.0, 128.0 / 255.0)  # navy
ra9 = (0.0, 100.0 / 255.0, 0.0)  # darkgreen
colors = [ra1, ra2, ra3, ra4, ra5, ra6, ra7, ra8, ra9]
bg_colors = [(1.0, 1.0, 1.0)]
action_names = [
    "COC",
    "DNC",
    "DND",
    "DES1500",
    "CL1500",
    "SDES1500",
    "SCL1500",
    "SDES2500",
    "SCL2500",
]

MEAN = np.array([0.0, 20.0])
RANGES = np.array([16000.0, 40.0])


def plot_loss(history):
    plt.figure()
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_predictions(model, xs, ys, xlim=[0, 60], ylim=[0, 60]):
    # neural network values
    y_predict = model.predict(xs)
    # scipy values
    popt, _ = scipy.optimize.curve_fit(lambda x, b0, b1: b0 + b1 * x, xs, ys)
    y_scipy = xs * popt[1] + popt[0]

    plt.figure()
    plt.plot(xs, y_predict)
    plt.scatter(xs, ys, color="C1")
    plt.plot(xs, y_scipy, color="C2")
    plt.legend(["predictions", "data", "scipy"])
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()


def plot_intervals(
    input_interval,
    output_interval,
    xs=None,
    ys=None,
    y_predict=None,
    y_scipy=None,
    xlim=[0, 60],
    ylim=[0, 60],
    desired_interval=None,
):
    fig = plt.figure()
    ax = fig.gca()

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if xs is not None:
        legend = []
        if ys is not None:
            plt.scatter(xs, ys, color="C2")
            legend.append("data")
        if y_predict is not None:
            plt.plot(xs, y_predict)
            legend.append("NN")
        if y_scipy is not None:
            plt.plot(xs, y_scipy, color="C1")
            legend.append("OLS")
        plt.legend(legend)

    input_width = input_interval[0].sup - input_interval[0].inf
    if type(output_interval[0]) == interval:
        output_width = output_interval[0][0].sup - output_interval[0][0].inf
    elif type(output_interval[0]) == list:
        output_width = output_interval[0][1] - output_interval[0][0]
    interval_rect = matplotlib.patches.Rectangle(
        (input_interval[0].inf, output_interval[0][0].inf), input_width, output_width
    )
    ax.add_collection(
        matplotlib.collections.PatchCollection(
            [interval_rect], facecolor="k", alpha=0.1, edgecolor="k"
        )
    )
    if desired_interval is not None:
        out_rect = matplotlib.patches.Rectangle(
            (-60, desired_interval[0].inf),
            120,
            desired_interval[0].sup - desired_interval[0].inf,
        )
        ax.add_collection(
            matplotlib.collections.PatchCollection(
                [out_rect], facecolor="r", alpha=0.1, edgecolor="r"
            )
        )

    plt.show()


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
        assert len(ivls) == 2
    new_ivls = []
    for i in range(2):
        new_ivls.append((ivls[i] - MEAN[i]) / RANGES[i])
    return new_ivls


def plot_policy(
    model: tf.keras.Model,
    filename: str = None,
    savefig: bool = False,
    zoom: bool = False,
    vo: float = 0,
    vi: float = 0,
    use_sisl_colors: bool = False,
    intervals: list = None,
    intervalcolor: str = None,
    title: str = None,
):
    x_grid = None
    taus = np.linspace(0, 40, 81)
    hs = np.hstack(
        [
            np.linspace(-5000, -2000, 20),
            np.linspace(-2000, 2000, 80),
            np.linspace(2000, 5000, 20),
        ]
    )
    for tau in taus:
        grid_component = np.vstack(
            [
                hs,
                np.ones(hs.shape) * tau,
            ]
        ).T
        if x_grid is not None:
            x_grid = np.vstack([x_grid, grid_component])
        else:
            x_grid = grid_component
    y_pred = model.predict(normalize_point(x_grid))
    # y_pred = model.predict(x_grid)
    advisory_idxs = np.argmax(y_pred, axis=1)

    # dict indexed by color/advisory of all points
    xs = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    ys = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}

    for i, advisory_idx in enumerate(advisory_idxs):
        scatter_x = x_grid[i, -1]  # tau
        scatter_y = x_grid[i, 0]  # h
        xs[advisory_idx].append(scatter_x)
        ys[advisory_idx].append(scatter_y)

    fig = plt.figure()
    plt.tight_layout()
    for i in range(len(colors)):
        if use_sisl_colors:
            plt.scatter(xs[i], ys[i], s=10, c=[colors[i]])
        else:
            plt.scatter(xs[i], ys[i], s=10)

    # add intervals
    if intervals is not None:
        ax = fig.gca()
        tau_interval = intervals[-1]  # x-coord
        h_interval = intervals[0]  # y-coord
        out_rect = matplotlib.patches.Rectangle(
            (tau_interval[0].inf, h_interval[0].inf),  # lower left anchor
            tau_interval[0].sup - tau_interval[0].inf,  # width
            h_interval[0].sup - h_interval[0].inf,  # height
            facecolor=intervalcolor if intervalcolor is not None else "b",
            alpha=0.3,
            edgecolor=intervalcolor if intervalcolor is not None else "b",
        )
        ax.add_patch(out_rect)

    plt.legend(action_names + ["Input Region"], loc="upper right")

    plt.xlabel("Tau (sec)")
    plt.ylabel("h (ft)")
    if title is None:
        plt.title(f"Policy for vo:{vo} and vi:{vi}")
    else:
        plt.title(title)
    if savefig:
        if filename is None:
            plt.savefig(f"images/viz_policy_vo{vo}_vi{vi}.pdf")
        else:
            if filename[-4:] == ".pdf":
                plt.savefig(filename)
            else:
                plt.savefig(filename + ".pdf")
    plt.show()

    if zoom:
        fig = plt.figure()
        ax = fig.gca()
        plt.tight_layout()
        for i in range(len(colors)):
            if use_sisl_colors:
                plt.scatter(xs[i], ys[i], s=10, c=[colors[i]])
            else:
                plt.scatter(xs[i], ys[i], s=10)
        # add intervals
        if intervals is not None:
            ax = fig.gca()
            tau_interval = intervals[-1]  # x-coord
            h_interval = intervals[0]  # y-coord
            out_rect = matplotlib.patches.Rectangle(
                (tau_interval[0].inf, h_interval[0].inf),  # lower left anchor
                tau_interval[0].sup - tau_interval[0].inf,  # width
                h_interval[0].sup - h_interval[0].inf,  # height
                facecolor=intervalcolor if intervalcolor is not None else "b",
                alpha=0.3,
                edgecolor=intervalcolor if intervalcolor is not None else "b",
            )
            ax.add_patch(out_rect)
            plt.xlim([0, 40])

        plt.legend(action_names + ["Input Region"], loc="upper right")
        plt.xlabel("Tau (sec)")
        plt.ylabel("h (ft)")
        if title is None:
            plt.title(f"Zoomed Policy for vo:{vo} and vi:{vi}")
        else:
            plt.title(f"Zoomed {title}")
        plt.ylim([-1100, 1100])
        if savefig:
            if filename is None:
                plt.savefig(f"images/viz_policy_vo{vo}_vi{vi}_zoomed.pdf")
            else:
                if filename[-4:] == ".pdf":
                    plt.savefig(filename[:-4] + "_zoomed.pdf")
                else:
                    plt.savefig(filename + "_zoomed.pdf")
        plt.show()
