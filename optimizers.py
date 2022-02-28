import numpy as np
from utils import deepsubtract, deepmultiply
import warnings

warnings.filterwarnings("ignore")


def update_step_size(epoch, c=3):
    """
    Update the step size according to the following expression: alpha = c / (k + 2)^s

    Arguments:
        epoch -- Epoch of training

    Keyword Arguments:
        c -- Initial learning rate (default: {3})

    Returns:
        Learning rate
    """
    s = 0.9
    return c / (epoch + 2)**s


def random_reshuffling(initial_x, num_of_epochs, optimized_function):
    """
    Implementation of the Random reshuffling algorithm.
            @article{,
               title={Why random reshuffling beats stochastic gradient descent},
               volume={186},
               ISSN={1436-4646},
               url={http://dx.doi.org/10.1007/s10107-019-01440-w},
               DOI={10.1007/s10107-019-01440-w},
               number={1-2},
               journal={Mathematical Programming},
               publisher={Springer Science and Business Media LLC},
               author={Gürbüzbalaban, M. and Ozdaglar, A. and Parrilo, P. A.},
               year={2019},
               month={Oct},
               pages={49–84}
            }

    Arguments:
        initial_x -- Initial x
        num_of_epochs -- Number of epochs
        optimized_function -- Function to be optimized

    Returns:
        Array of results of training
    """
    xs = [initial_x]
    objectives = [optimized_function.objective(initial_x)]
    x = initial_x

    num_of_components = optimized_function.num_of_functions

    for epoch in range(num_of_epochs):

        permutation = np.random.permutation(np.arange(num_of_components))

        ss = update_step_size(epoch, optimized_function.c)

        for i in range(num_of_components):

            gradient = optimized_function.gradient(x, permutation[i])
            x = deepsubtract(x, deepmultiply(gradient, ss))

        objective = optimized_function.objective(x)
        objectives.append(objective)
        xs.append(x)

    return np.array(xs), np.array(objectives)


def SGD(initial_x, num_of_epochs, optimized_function):
    """
    Implementation of the Stochastic gradient descent algorithm.
            @Book{,
              author = {Poliak, B. T.},
              title = {Introduction to optimization},
              publisher = {New York: Optimization Software, Publications Division},
              year = 1987
            }

    Arguments:
        initial_x -- Initial x
        num_of_epochs -- Number of epochs
        optimized_function -- Function to be optimized

    Returns:
        Array of results of training
    """
    xs = [initial_x]
    objectives = [optimized_function.objective(initial_x)]
    x = initial_x

    num_of_components = optimized_function.num_of_functions

    for epoch in range(num_of_epochs):

        ss = update_step_size(epoch, optimized_function.c)

        for i in range(num_of_components):

            index = np.random.choice(num_of_components, 1)[0]

            gradient = optimized_function.gradient(x, index)
            x = deepsubtract(x, deepmultiply(gradient, ss))

        objective = optimized_function.objective(x)
        objectives.append(objective)
        xs.append(x)

    return np.array(xs), np.array(objectives)