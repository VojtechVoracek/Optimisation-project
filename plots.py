
import matplotlib.pyplot as plt
from optimized_functions import *
from optimizers import *
import warnings
from tqdm import tqdm

plt.rcParams['text.usetex'] = True

def calc_x_q_k(xs: np.array, q: float):
    """
    calculates the average of last q*k iterates for epoch k
    Args:
        xs (np.array): x values optained from optimization
        q (float): between 0 and 1 to give ratio of last k iterates to be used
    """
    t = len(xs)
    x_q_k = [sum(xs[int(np.ceil((1-q)*k)) :k+1])/(k+1-np.ceil((1-q)*k)) for k in range(t)]
    return(np.array(x_q_k))

def process_run(optimized_function, num_of_epochs, x_0, tol=-1e-7, plot_average=False, q=None):
    """
        Processes one run of SGD and RR algorithms parameter function
    :param optimized_function: Class
            Unconstrained, finite-sum function to optimize.
    :param num_of_epochs: int
            Maximal number of epochs for both algorithms.
    :param x_0: ndarray or scalar
            Initial point
    :param tol: float
            Tolerance, epsilon.
    :param plot_average:
            If True: plot the distance of q-suffix average \overline{x}_{q,k} to the optimal solution x^*
            If False plot the distance of the current x to the optimal solution x^*
    :param q:
            The q-suffix parameter. Define the number of last iterates from which the average is computed.
    :return: scg_xs, rr_xs
    """
    scg_xs, scg_objectives = SGD(x_0, num_of_epochs, optimized_function, tol)
    rr_xs, rr_objectives = random_reshuffling(x_0, num_of_epochs, optimized_function, tol)

    if plot_average:
        # Change iterates to the q-suffix average
        scg_xs = calc_x_q_k(scg_xs, q)
        rr_xs = calc_x_q_k(rr_xs, q)

    # calculate distance to the optimal solution
    scg_xs = np.linalg.norm(scg_xs - optimized_function.x_star, axis=1)
    rr_xs = np.linalg.norm(rr_xs - optimized_function.x_star, axis=1)

    scg_xs = scg_xs[np.where(scg_xs >= tol)]
    rr_xs = rr_xs[np.where(rr_xs >= tol)]

    return scg_xs, rr_xs
    
def plot_runs(optimized_function, num_of_epochs, num_of_runs, tol=-1e-7, plot_average=False, q=None, subplots = [], fixed_plot_k=1):
    """
            Create graphs capturing the performance of SGD and RR algorithms on the sphere function
        :param optimized_function: Class
                Unconstrained, finite-sum function to optimize.
        :param num_of_epochs: int
                Maximal number of epochs for both algorithms.
        :param num_of_runs: int
                Number of independent runs of both algorithms.
        :param tol: float
                Tolerance, epsilon.
        :param plot_average:
                If True: plot the distance of q-suffix average \overline{x}_{q,k} to the optimal solution x^*
                If False plot the distance of the current x to the optimal solution x^*
        :param q:
                The q-suffix parameter. Define the number of last iterates from which the average is computed.
        :param subplots: (d, ) int
                Array of settings to the function. Currently only for dimensions of the sphere function.        
        :param fixed_plot_k: int
                Constant to move the O(1/k*s) rate curve. For plotting purposes only.
        :return: None
    """

    s=0.9
    has_subplots = len(subplots) > 0

    #verify. currently only spheres function allows custom settings to plot on multiple subplots.
    if(has_subplots and optimized_function.name!="Square"):
        warnings.warn('Skipping subplots. Please define its behaviour for current function.')
        subplots = []
        has_subplots = False


    if(has_subplots):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        axs = [ax1, ax2, ax3, ax4]
        linewidth = 0.5
    else:
        fig, ((ax1)) = plt.subplots(1, 1)
        axs = [ax1]
        linewidth = 0.1
    

    if(optimized_function.name=="LeastSquares"):
        average_len = num_of_epochs * optimized_function.A.shape[0] + 1
    elif (optimized_function.name=="Square"):
        average_len = num_of_epochs + 1
    else:
        average_len = num_of_epochs + 1

    
    settings = subplots if has_subplots else [1] # dummy setting for unique test

    for d in range(len(settings)):

        dimension = settings[d]

        scg_average = np.zeros(average_len)
        rr_average = np.zeros(average_len)

        if(has_subplots):
            if(optimized_function.name=="Square"): #update settings
                optimized_function.set_dimension(dimension)
            else:
                warnings.warn('New settings ommited. Please define its behaivour for current function.')

        if(optimized_function.name=="LeastSquares"):
            x_0_size = optimized_function.A.shape[1]
        elif (optimized_function.name=="Square" or optimized_function.name == "Non Convex Function"):
            x_0_size = optimized_function.num_of_functions #dimension
        else:
            x_0_size = 1

        for i in tqdm(range(num_of_runs)):
            x_0 = np.random.uniform(low=-10, high=10, size=x_0_size)
            scg_xs, rr_xs = process_run(optimized_function, num_of_epochs, x_0, tol, plot_average, q)

            # plot individual runs
            axs[d].plot(scg_xs, linewidth=linewidth, color="dodgerblue")
            axs[d].plot(rr_xs, linewidth=linewidth, color="lightcoral")

            # add this run to average over all runs
            scg_average[:len(scg_xs)] = scg_average[:len(scg_xs)] + scg_xs
            rr_average[:len(rr_xs)] = rr_average[:len(rr_xs)] + rr_xs

        scg_average = scg_average / num_of_runs
        rr_average = rr_average / num_of_runs

        scg_average = scg_average[np.where(scg_average >= tol)]
        rr_average = rr_average[np.where(rr_average >= tol)]


        axs[d].set_yscale('log')
        if d % 2 == 1:
            axs[d].yaxis.tick_right()
        if d >= 2 or not has_subplots:
            axs[d].set_xlabel("Epoch")

        if plot_average:
            axs[d].set_ylabel(r'$||\overline{x}_{q,k} - x^{*}||$')
        else:
            axs[d].set_ylabel(r'$||x - x^{*}||$')


        # make the O(1 / k^s) ratio curve
        k = np.arange(len(scg_average))
        ks = fixed_plot_k / (k + 1) ** s
    
        axs[d].legend([axs[d].plot(scg_average, color="blue")[0], axs[d].plot(rr_average, color="red")[0],
                       axs[d].plot(k, ks, color="black", linestyle="--")[0]],
                      ["SGD", "RR", r'$\mathcal{O}(1 / k^{s})$'])

        if(has_subplots):
            subtitle = "d=" + str(dimension)
            axs[d].set_title(subtitle, y=0.8, x=0.42)
    
    fig.suptitle(optimized_function.plot_title, size=20)

    if plot_average:
        fig.savefig(optimized_function.name + "_runs_average.png")
    else:
        fig.savefig(optimized_function.name + "_runs.png")