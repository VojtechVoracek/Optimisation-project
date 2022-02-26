
import string
import matplotlib.pyplot as plt
import numpy as np
from utils import padded_list

plt.rcParams['text.usetex'] = True


def plot_xs(sgd_xs: list, rr_xs: list, s: float, name: str, plot_to=False):
    """
    Plot results of single experiment

    Arguments:
        sgd_xs -- Results of SGD run
        rr_xs -- Results of RR run
        s -- Parameter s
        name -- Title of plot

    Keyword Arguments:
        plot_to -- Whether to plot to subplot or not. If yes the subplot to be plotted to (default: {False})
    """
    if plot_to:
        ax = plot_to
        fig = False
    else:
        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel("Epoch")
    linewidth = 0.1
    ax.set_yscale('log')
    ax.set_ylabel(r'$||\overline{x}_{q,k} - x^{*}||$')

    sgd_xs = padded_list(sgd_xs)
    rr_xs = padded_list(rr_xs)

    no_epochs = max(max([len(i) for i in sgd_xs]),
                    max([len(i) for i in rr_xs]))
    no_runs = len(sgd_xs)

    sgd_average = np.mean(np.vstack(sgd_xs), axis=0)
    rr_average = np.mean(np.vstack(rr_xs), axis=0)

    # make the O(1 / k^s) ratio curve
    k = np.arange(no_epochs)
    ks = max(sgd_average[0], rr_average[0]) / (k + 1) ** s

    for i in range(no_runs):
        ax.plot(sgd_xs[i], linewidth=linewidth, color="dodgerblue")
        ax.plot(rr_xs[i], linewidth=linewidth, color="lightcoral")

    ax.legend([ax.plot(sgd_average, color="blue")[0], ax.plot(rr_average, color="red")[0],
               ax.plot(k, ks, color="black", linestyle="--")[0]],
              ["SGD", "RR", r'$\mathcal{O}(1 / k^{s})$'])

    if fig:
        fig.suptitle(name, size=20)

        fig.savefig(name + "_run.png")


def plot_with_subplots(sgd_xs_list: list, rr_xs_list: list, s: float, name: str, subnames: list):
    """
    Plot multiple experiments in one plot

    Arguments:
        sgd_xs_list -- List of results of SGD experiments
        rr_xs_list -- List of results of RR experiments
        s -- Parameter s
        name -- Title of plot
        subnames -- descriptions of subplots
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    axs = [ax1, ax2, ax3, ax4]
    for i in range(4):
        plot_xs(sgd_xs_list[i], rr_xs_list[i], s, subnames[i], axs[i])
        subtitle = "d=" + subnames[i]
        axs[i].set_title(subtitle, y=0.8, x=0.42)
    axs[1].yaxis.tick_right()
    axs[3].yaxis.tick_right()
    axs[2].set_xlabel("Epoch")
    axs[3].set_xlabel("Epoch")
    fig.savefig(name + "_runs.png")
