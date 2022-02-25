import numpy as np
from Neural_Network import MyMLP, Leaky_ReLU
from utils import deepmultiply

class ComponentFunction:
    """
        A class to represent the Component function.

        Attributes
        ----------
        optimal_value : int
                The minimum of the function.
        num_of_functions : int
                Number of f_i within the sum. f(x) = f_1(x) + f_2(x) + ... + f_num_of_functions(x)
        name : string
                Name of the function when plotted.
        x_star : [1, 1] float
                The minimizer of the function.

        Methods
        -------
        objective(x):
                Returns the value of function at point x.

        gradient(projection):
                Returns the gradient of function at point x.
    """
    def __init__(self):
            
        self.optimal_value = 1
        self.num_of_functions = 2
        self.name = "ComponentFunction"
        self.plot_title = "Component function"
        self.c=1
        self.x_star = np.array([0])
        
    def x_0(self):
        return(np.random.uniform(low = -10, high = 10, size = 1))

    def objective(self, x):
        """
            Returns the value of function at point x.
            f(x) = 0.5 * x^2 + 1

            :param x:   [1, 1] float
            :return:    float
        """
        return 1.5 * x[0]**2 + 1

    def gradient(self, x, index):
        """
            Returns the gradient of function at point x.
            df_1(x) = x - 1
            df_2(x) = 2x + 1

            :param x:   [1, 1] float
            :return:    [1, 1] float
        """
        if index == 0:
            return x - 1
        if index == 1:
            return 2 * x + 1


class SphereFunction:
    """
        A class to represent the Sphere function.

        Attributes
        ----------
        optimal_value : int
                The minimum of the function.
        num_of_functions : int
                Number of f_i within the sum. f(x) = f_1(x) + f_2(x) + ... + f_num_of_functions(x)
        name : string
                Name of the function when plotted.
        x_star : (num_of_functions, ) float
                The minimizer of the function.

        Methods
        -------
        objective(x):
                Returns the value of function at point x.

        gradient(projection):
                Returns the gradient of function at point x.
    """
    def __init__(self, dimension):
        self.dimension = dimension
        self.optimal_value = 0
        self.num_of_functions = self.dimension
        self.name = "Square"
        self.plot_title = "Sphere function"
        self.c=1
        self.x_star = np.zeros(self.dimension)
        
    def x_0(self):
        return(np.random.uniform(low = -10, high = 10, size = self.dimension))

    def objective(self, x):
        """
            Returns the value of function at point x.
            f(x) = x_1^2 + x_2^2 + ... + x_d^2

            :param x:   (self.dimension, ) float
            :return:    float
        """
        return np.linalg.norm(x, 2) ** 2

    def gradient(self, x, index):
        """
            Returns the gradient of function at point x.
            df_i(x) = [0,...,0, 2x_i, 0,..0]

            :param x:   (self.dimension, ) float
            :return:    (self.dimension, ) float
        """
        grad = np.zeros_like(x)
        grad[index] = 2 * x[index]
        return grad


class LinearRegression:
    """
        A class to represent the Linear regression problem.

        Attributes
        ----------
        optimal_value : int
                The minimum of the function.
        num_of_functions : int
                Number of f_i within the sum. f(x) = f_1(x) + f_2(x) + ... + f_num_of_functions(x)
        name : string
                Name of the function when plotted.
        x_star : (d, ) (float,)
                The minimizer of the function.
        A : [num_of_functions, d] float
                The data matrix.
        b : (num_of_functions,) float
                The target vector.

        Methods
        -------
        objective(x):
                Returns the value of function at point x.

        gradient(projection):
                Returns the gradient of function at point x.
    """
    optimal_value = 0
    num_of_functions = None
    name = "LeastSquares"
    plot_title = "Linear regression - diabetes dataset"

    def __init__(self, A, b):
        self.num_of_functions = A.shape[0]
        self.A = A
        self.b = b[:, 0]
        self.x_star = (np.linalg.inv(A.T @ A) @ A.T @ b)[:, 0]
        self.optimal_value = self.objective(self.x_star)
        self.name = "LeastSquares"
        self.plot_title = "Linear regression - diabetes dataset"
        self.c=0.1
        
    def x_0(self):
        return(np.random.uniform(low = -10, high = 10, size = self.A.shape[1]))

    def objective(self, x):
        """
            Returns the value of function at point x.
            f(x) = ||Ax - b||^2

            :param x:   (float, ) (self.dimension, )
            :return:    float
        """
        return np.linalg.norm(self.A @ x - self.b) ** 2

    def gradient(self, x, index):
        """
            Returns the gradient of function at point x.
            df_i(x) = 2 * a_i (a_i^T x - b_i)
            :param x:   (float, ) (self.dimension, )
            :return:    (float, ) (self.dimension, )
        """
        grad = 2 * self.A[index, :].T * (self.A[index, :] @ x - self.b[index])
        return grad

class Neural_Network:
    """
        A class to represent a Neural Network.

        Attributes
        ----------
        optimal_value : int
                The minimum of the function.
        name : string
                Name of the function when plotted.

        Methods
        -------
        objective(x):
                Returns the value of function at point x.

        gradient(projection):
                Returns the gradient of function at point x.
    """

    def __init__(self, A, b, hidden_layer_sizes=[50,40], activations=[Leaky_ReLU, Leaky_ReLU]):
        self.num_of_functions = A.shape[0]
        self.A = A
        self.b = b[:, 0]
        self.optimal_value = 0
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activations = activations 
        self.name = "Neural Network"
        self.plot_title = "Neural Network"
        self.c=0.003
        
    def x_0(self):
        self.make_nn()
        self.x_star = [deepmultiply(self.nn.weights, 0), deepmultiply(self.nn.biases, 0)]  
        return([self.nn.weights, self.nn.biases])
        
    def make_nn(self):
        self.nn = MyMLP(self.hidden_layer_sizes, self.activations)
        self.nn.initialize_weights(self.A)

    def objective(self, x):
        """
            Returns the value of the NN at point x

            :param x:   (self.dimension, ) float
            :return:    float
        """
        temp_weights = self.nn.weights.copy()
        temp_biases = self.nn.biases.copy()
        
        self.nn.weights = x[0]
        self.nn.biases = x[1]
        objective = self.nn.mean_squared_error(self.A, self.b)
        
        self.nn.weights = temp_weights
        self.nn.biases = temp_biases
        return objective

    def gradient(self, x, index):
        """
            Returns the gradient of function at point x.
            :param x:   (self.dimension, ) float
            :return:    (self.dimension, ) float
        """
        self.nn.weights = x[0]
        self.nn.biases = x[1]
        weight_grad, bias_grad = self.nn.grad(np.expand_dims(self.A[index, :], 1).T, self.b[index])
        return [weight_grad, bias_grad]