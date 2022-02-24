import numpy as np
from optimizers import SGD, random_reshuffling
from utils import calc_x_q_k
from tqdm import tqdm

def process_run(optimized_function, num_of_epochs, return_average=False, q=None, return_objectives = False):
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
    :return: sgd_xs, rr_xs
    """
    x_0 = optimized_function.x_0()
    
    sgd_values, sgd_objectives = SGD(x_0, num_of_epochs, optimized_function)
    rr_values, rr_objectives = random_reshuffling(x_0, num_of_epochs, optimized_function)
    
    if return_objectives == True:
        sgd_xs = sgd_objectives
        sgd_xs = sgd_xs.reshape(-1, 1)
        rr_xs = rr_objectives
        rr_xs = rr_xs.reshape(-1, 1)
    else:
        sgd_xs = sgd_values
        rr_xs = rr_values

    if return_average:
        # Change iterates to the q-suffix average
        sgd_xs = calc_x_q_k(sgd_xs, q)
        rr_xs = calc_x_q_k(rr_xs, q)

    # calculate distance to the optimal solution
    if return_objectives == False:
        sgd_xs = sgd_xs - optimized_function.x_star
        rr_xs = rr_xs - optimized_function.x_star
    
    sgd_xs = np.linalg.norm(sgd_xs, axis=1)
    rr_xs = np.linalg.norm(rr_xs, axis=1)

    return sgd_xs, rr_xs

def simulate_runs(optimized_function, num_of_epochs, num_of_runs, q=None, return_objectives = False):
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

    sgd_list = []
    rr_list = []

    for _ in tqdm(range(num_of_runs)):
        sgd_xs, rr_xs = process_run(optimized_function, num_of_epochs, True, q, return_objectives)
        sgd_list.append(sgd_xs)
        rr_list.append(rr_xs)
        
    return(sgd_list, rr_list)