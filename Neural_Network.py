import numpy as np

def relu(t, gradient=False):
    """Rectified linear unit activation function.

    Parameters
    ----------
    t : numpy array
        pre-activations
    gradient : bool, optional
        A flag used to indicate whether the gradient of the activation should
        be computed.

    Returns
    -------
    activations : numpy array
        output after applying the activation function
    gradient : numpy array
        gradient of activation function. Only returned if flag is set to true.
    """
    t = np.maximum(0, t)
    if not gradient:
        return t
    else:
        return t, (t > 0).astype(float)

def Leaky_ReLU(t, gradient = False):
    if not gradient:
        t = np.where(t>0, t, 0.1*t)
        return(t)
    else:
        t = np.where(t>0, t, 0.1*t)
        grad = np.where(t>0, 1, 0.1)
        return t, grad
    

def identity(t, gradient=False):
    """Identity activation function.

    Parameters
    ----------
    t : numpy array
        pre-activations
    gradient : bool, optional
        A flag used to indicate whether the gradient of the activation should
        be computed.

    Returns
    -------
    activations : numpy array
        output after applying the activation function
    gradient : numpy array
        gradient of activation function. Only returned if flag is set to true.
    """
    if not gradient:
        return t
    else:
        return t, np.ones_like(t)

class MyMLP():
    def __init__(self, hidden_layer_sizes=[10,10], activations=[relu, relu]):
        assert len(activations) == len(hidden_layer_sizes), "Invalid number of layers/activations."

        self.hidden_layer_sizes = hidden_layer_sizes.copy()
        self.hidden_layer_sizes.append(1) # single output at final layer
        self.activations = activations.copy()
        self.activations.append(identity)
        self.cache_post_activations = []
        self.cache_derivatives = []
        
    def initialize_weights(self, x):
        self.weights = []
        self.biases = []
        for i in range(len(self.hidden_layer_sizes)):
            if i == 0:
                dim1 = x.shape[1]
            else:
                dim1 = self.hidden_layer_sizes[i-1]
            dim2 = self.hidden_layer_sizes[i]
            if (self.activations[i] == relu) or (self.activations[i] == identity) or (self.activations[i] == Leaky_ReLU): 
                W =  np.sqrt(2. / (dim1 + dim2)) *  np.random.randn(dim2, dim1)
            else:
                raise ValueError("No initialization for layer with activation %s" % self.activations[i])
            b = np.zeros(dim2)

            self.weights.append(W)
            self.biases.append(b)

    def grad(self, x, y):
        weight_gradient = []
        bias_gradient = []
        _, grad = self.mean_squared_error(x, y, cache=True, gradient=True)
        #Gradient of output layer
        weight_gradient.append(grad.T.dot(self.cache_post_activations[-2]))
        bias_gradient.append(grad.sum(axis=0))
        #Gradient of hidden layers via backprop
        for i in range(1, len(self.hidden_layer_sizes)):
            grad = self.cache_derivatives[-i-1]*(grad.dot(self.weights[-i]))
            weight_gradient.append(grad.T.dot(self.cache_post_activations[-i-2]))
            bias_gradient.append(grad.T.sum(axis=1))
        weight_gradient.reverse()
        bias_gradient.reverse()
        return weight_gradient, bias_gradient

    def mean_squared_error(self, x, y, cache=False, gradient=False):
        """Mean squared error loss function.

        Parameters
        ----------
        x : numpy array
            Input data (shape n_samples x n_features)
        y : numpy array
            Targets (shape n_samples x 1)   
        cache : bool, optional
            If set to true, caches activations and derivatives of activation
            functions for usage during backprop.
        gradient : bool, optional
            A flag used to indicate whether the gradient of the activation should
            be computed.

        Returns
        -------
        err : float
            Squared loss.
        g : numpy array
            gradient of loss function
        """
        t = self.predict(x, cache=cache)
        err = np.mean((y - t)**2)
        if not gradient:
            return err
        else:
            g =  -2 * (y - t)
            return err, g

    def predict(self, x, cache=False):
        """Compute predictions for a batch of samples.

        Parameters
        ----------
        x : numpy array
            Input data (shape n_samples x n_features)
        cache : bool, optional
            If set to true, caches activations and derivatives of activation
            functions for usage during backprop.

        Returns
        -------
        t : numpy array
            Predictions.
        """
        if cache:
            self.cache_post_activations.append(x.copy())
        t = x.T
        for i in range(len(self.hidden_layer_sizes)):
            t = np.matmul(self.weights[i], t) + np.expand_dims(self.biases[i], 1)
            if not cache:
                t = self.activations[i](t)
            else:
                t, g = self.activations[i](t, gradient=True)

            if cache:
                self.cache_post_activations.append(t.T)
                self.cache_derivatives.append(g.T)

        return t.T
