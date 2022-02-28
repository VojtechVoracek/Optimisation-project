from optimized_functions import *
from plots import plot_xs, plot_with_subplots
from run import simulate_runs
from sklearn.datasets import load_diabetes

def sphere_function_run():
    """
    Experiment for Sphere function
    """
    max_epochs = 50
    q = 0.2
    runs = 100
    s = 0.9
    
    sgd_xs_list = []
    rr_xs_list = []
    dims = [2, 3, 5, 10]
    
    for i in dims:
        sgd_xs, rr_xs = simulate_runs(SphereFunction(i), max_epochs, runs, q, False)
        sgd_xs_list.append(sgd_xs)
        rr_xs_list.append(rr_xs)
    plot_with_subplots(sgd_xs_list, rr_xs_list, s, "Sphere function", [str(i) for i in dims])
    
def component_function_run():
    """
    Experiment for component function
    """
    max_epochs = 500
    runs = 100
    q = 0.2
    s = 0.9
    
    sgd_xs, rr_xs = simulate_runs(ComponentFunction(), max_epochs, runs, q, False)
    plot_xs(sgd_xs, rr_xs, s, "Component Function")
    
def linear_regression_run():
    """
    Experiment for linear regression
    """
    max_epochs = 2000
    runs = 25
    q = 0.2
    s = 0.9
    
    data = load_diabetes()
    A = data['data']
    A = (A - np.mean(A, axis = 0))/np.std(A, axis = 0)
    b = np.expand_dims(data['target'], 1)
    b = (b - np.mean(b, axis = 0))/np.std(b, axis = 0)
    
    sgd_xs, rr_xs = simulate_runs(LinearRegression(A,b), max_epochs, runs, q, False)
    plot_xs(sgd_xs, rr_xs, s, "Linear Regression")
    
def nn_run():
    """
    Experiment for Neural Network
    """
    max_epochs = 1000
    runs = 20
    q = 0.2
    s = 0.3
    
    data = load_diabetes()
    A = data['data']
    A = (A - np.mean(A, axis = 0))/np.std(A, axis = 0)
    b = np.expand_dims(data['target'], 1)
    b = (b - np.mean(b, axis = 0))/np.std(b, axis = 0)
    
    sgd_xs, rr_xs = simulate_runs(Neural_Network(A,b), max_epochs, runs, q, True)
    plot_xs(sgd_xs, rr_xs, s, "Neural Network")