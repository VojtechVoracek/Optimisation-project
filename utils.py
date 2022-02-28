import numpy as np


def deepmultiply(x, a):
    """
    Multiply all values contained in x with a

    Arguments:
        x -- List or Array or float
        a -- float

    Returns:
        x*a
    """
    if isinstance(x, list):
        return [deepmultiply(i, a) for i in x]

    if isinstance(x, type(np.array(None))):
        return x*a

    return x*a


def deepsubtract(x, y):
    """
    Subtract all elements in y from the according elements in x

    Arguments:
        x -- List or array or float
        y -- List or array or float

    Returns:
        x-y
    """
    if isinstance(x, list):
        return [deepsubtract(x[i], y[i]) for i in range(len(x))]

    if isinstance(x, type(np.array(None))):
        return x-y

    return x-y


def padded_list(arraylist):
    """
    Extends arrays with zeros so all arrays have the same length

    Arguments:
        arraylist -- List of arrays
    """
    outarr = np.zeros((len(arraylist), np.max([len(ps) for ps in arraylist])))
    for i, c in enumerate(arraylist):
        outarr[i, :len(c)] = c
    return([outarr[i] for i in range(len(outarr))])


def calc_x_q_k(xs: np.array, q: float):
    """
    calculates the average of last q*k iterates for epoch k

    Arguments:
        xs -- x values
        q -- Parameter q
    """
    t = len(xs)
    x_q_k = [sum(xs[int(np.ceil((1-q)*k)):k+1])/(k+1-np.ceil((1-q)*k))
             for k in range(t)]
    return(np.array(x_q_k))
