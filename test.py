from optimized_functions import *
from plots import plot_xs
from sklearn.datasets import load_diabetes
import numpy as np
from optimizers import gradient_descent
from utils import calc_x_q_k
from tqdm import tqdm

def nn_run():
    """
    Experiment for Neural Network
    """
    max_epochs = 100
    runs = 1
    q = 0.2
    s = 0.9
    
    data = load_diabetes()
    A = data['data']
    A = (A - np.mean(A, axis = 0))/np.std(A, axis = 0)
    b = np.expand_dims(data['target'], 1)
    b = (b - np.mean(b, axis = 0))/np.std(b, axis = 0)
    
    sgd_xs, rr_xs = simulate_runs(Neural_Network(A,b), max_epochs, runs, q, True)
    plot_xs(sgd_xs, rr_xs, s, "Neural Network")
    
def process_run(optimized_function, num_of_epochs, return_average=False, q=None, return_objectives=False):
    """
    Single run of SGD and RR for selected function

    Arguments:
        optimized_function -- Function to be optimized
        num_of_epochs -- Number of epochs of training

    Keyword Arguments:
        return_average -- Whether to return the q suffix average or not (default: {False})
        q -- Parameter q (default: {None})
        return_objectives -- Whether to return objectives or x values (default: {False})

    Returns:
        arrays of results for each epoch of SGD and RR
    """
    x_0 = optimized_function.x_0()

    sgd_values, sgd_objectives = gradient_descent(x_0, num_of_epochs, optimized_function)
    rr_values, rr_objectives = gradient_descent(x_0, num_of_epochs, optimized_function)

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


def simulate_runs(optimized_function, num_of_epochs, num_of_runs, q=None, return_objectives=False):
    """
    Run simulations

    Arguments:
        optimized_function -- Function to optimize
        num_of_epochs -- Number of epochs of training
        num_of_runs -- Number of runs

    Keyword Arguments:
        q -- Parameter q for moving average (default: {None})
        return_objectives -- Whether to return the objectives or the parameters that were optimized (default: {False})
    """
    sgd_list = []
    rr_list = []

    for _ in tqdm(range(num_of_runs)):
        sgd_xs, rr_xs = process_run(
            optimized_function, num_of_epochs, True, q, return_objectives)
        sgd_list.append(sgd_xs)
        rr_list.append(rr_xs)

    return(sgd_list, rr_list)

if __name__ == "__main__":
    nn_run()