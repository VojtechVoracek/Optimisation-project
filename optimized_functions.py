import numpy as np


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

    optimal_value = 1
    num_of_functions = 2
    name = "ComponentFunction"
    x_star = np.array([0])

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
    optimal_value = 0
    num_of_functions = None
    name = "Square"
    x_star = None

    def __init__(self, dimension):
        self.num_of_functions = dimension
        self.x_star = np.zeros(dimension)

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
    x_star = None
    A = None
    b = None

    def __init__(self, A, b):
        self.num_of_functions = A.shape[0]
        self.A = A
        self.b = b[:, 0]
        self.x_star = (np.linalg.inv(A.T @ A) @ A.T @ b)[:, 0]
        self.optimal_value = self.objective(self.x_star)

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
