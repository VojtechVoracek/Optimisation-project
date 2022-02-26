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
        c : float
                Initial learning rate
        x_star : [1, 1] float
                The minimizer of the function.

        Methods
        -------
        x_0():
                Initialize x.

        objective(x):
                Returns the value of function at point x.

        gradient(projection):
                Returns the gradient of function at point x.
    """

    def __init__(self):
        self.optimal_value = 1
        self.num_of_functions = 2
        self.c = 1.
        self.x_star = np.array([0])

    def x_0(self):
        """
            Initialize x.
        """
        return(np.random.uniform(low=-10, high=10, size=1))

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
        dimension : int
                Number of components of Sphere function        
        optimal_value : int
                The minimum of the function.
        num_of_functions : int
                Number of f_i within the sum. f(x) = f_1(x) + f_2(x) + ... + f_num_of_functions(x)
        c : float
                Initial learning rate
        x_star : (num_of_functions, ) float
                The minimizer of the function.

        Methods
        -------
        x_0():
                Initialize x.

        objective(x):
                Returns the value of function at point x.

        gradient(projection):
                Returns the gradient of function at point x.
    """

    def __init__(self, dimension):
        self.dimension = dimension
        self.optimal_value = 0
        self.num_of_functions = self.dimension
        self.c = 1.
        self.x_star = np.zeros(self.dimension)

    def x_0(self):
        """
        Initialize x.
        """
        return(np.random.uniform(low=-10, high=10, size=self.dimension))

    def objective(self, x):
        """
        Returns the value of function at point x.
        f(x) = x_1^2 + x_2^2 + ... + x_d^2

        Arguments:
            x -- Array of x values

        Returns:
            Squared norm of x
        """
        return np.linalg.norm(x, 2) ** 2

    def gradient(self, x, index):
        """
        Returns the gradient of function at point x.
        df_i(x) = [0,...,0, 2x_i, 0,..0]

        Arguments:
            x -- Array of x values
            index -- Index at which to optimize

        Returns:
            Gradient of f
        """
        grad = np.zeros_like(x)
        grad[index] = 2 * x[index]
        return grad


class LinearRegression:
    """
        A class to represent the Linear regression problem.

        Attributes
        ----------
        num_of_functions : int
                Number of f_i within the sum. f(x) = f_1(x) + f_2(x) + ... + f_num_of_functions(x)
        A : [num_of_functions, d] float
                The data matrix.
        b : (num_of_functions,) float
                The target vector.
        x_star : (d, ) (float,)
                The minimizer of the function.
        optimal_value : float
                The minimum of the function.
        c : float
                Initial learning rate.

        Methods
        -------
        x_0():
                Initialize x.

        objective(x):
                Returns the value of function at point x.

        gradient(projection):
                Returns the gradient of function at point x.
    """

    def __init__(self, A, b):
        self.num_of_functions = A.shape[0]
        self.A = A
        self.b = b[:, 0]
        self.x_star = (np.linalg.inv(A.T @ A) @ A.T @ b)[:, 0]
        self.optimal_value = self.objective(self.x_star)
        self.c = 0.1

    def x_0(self):
        """
        Initialize x.
        """
        return(np.random.uniform(low=-10, high=10, size=self.A.shape[1]))

    def objective(self, x):
        """
        Returns the value of function at point x.
        f(x) = ||Ax - b||^2

        Arguments:
            x -- Array of parameters of linear regresion

        Returns:
            Square error
        """
        return((np.linalg.norm(self.A @ x - self.b) ** 2)/self.num_of_functions)

    def gradient(self, x, index):
        """
        Returns the gradient of function at point x.
        df_i(x) = 2 * a_i (a_i^T x - b_i)

        Arguments:
            x -- Array of parameters of linear regression
            index -- sample at which to optimize

        Returns:
            Gradient
        """
        grad = 2 * self.A[index, :].T * (self.A[index, :] @ x - self.b[index])
        return grad


class Neural_Network:
    """
        A class to represent a Neural Network.

        Attributes
        ----------
        num_of_functions : int
                Number of f_i within the sum. f(x) = f_1(x) + f_2(x) + ... + f_num_of_functions(x)
        A : [num_of_functions, d] float
                The data matrix.
        b : (num_of_functions,) float
                The target vector.
        optimal_value : float
                Placeholder.
        hidden_layer_sizes: list
                List of sizes of hidden layers of NN.
        activations: list
                List of activation functions.
        c : float
                Initial learning rate.

        Methods
        -------
        x_0():
                Initialize x.

        make_nn():
                Create and initialize NN

        objective(x):
                Returns the value of function at point x.

        gradient(projection):
                Returns the gradient of function at point x.
    """

    def __init__(self, A, b, hidden_layer_sizes=[50, 40], activations=[Leaky_ReLU, Leaky_ReLU]):
        self.num_of_functions = A.shape[0]
        self.A = A
        self.b = b[:, 0]
        self.optimal_value = 0
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activations = activations
        self.c = 0.0003

    def x_0(self):
        """
        Initialize x.
        """
        self.make_nn()
        self.x_star = [deepmultiply(
            self.nn.weights, 0), deepmultiply(self.nn.biases, 0)]
        return([self.nn.weights, self.nn.biases])

    def make_nn(self):
        """
        Create and initialize NN.
        """
        self.nn = MyMLP(self.hidden_layer_sizes, self.activations)
        self.nn.initialize_weights(self.A)

    def objective(self, x):
        """
        Calculate objective of NN with weights and biases in x.

        Arguments:
            x -- List of weights and biases.

        Returns:
            Objective of NN.
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
        Returns gradient of nn with sampleat sample index.
        Arguments:
            x -- List of weights and biases.
            index -- Sample to be optimized

        Returns:
            Gradient of NN
        """
        self.nn.weights = x[0]
        self.nn.biases = x[1]
        weight_grad, bias_grad = self.nn.grad(
            np.expand_dims(self.A[index, :], 1).T, self.b[index])
        return [weight_grad, bias_grad]
