import nn
import math
import numpy as np


class LogisticRegressionModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new LogisticRegressionModel instance.

        A logistic regressor classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        Initialize self.w and self.alpha here
        self.alpha = *some number* (optional)
        self.w = []
        """
        "*** YOUR CODE HERE ***"
        self.alpha = 0.01
        self.w = [0 for i in range(dimensions)]

    def get_weights(self):
        """
        Return a list of weights with the current weights of the regression.
        """
        return self.w

    def DotProduct(self, w, x):
        """
        Computes the dot product of two lists
        Returns a single number
        """
        "*** YOUR CODE HERE ***"        
        return(np.dot(np.transpose(x), w))



    def sigmoid(self, x):
        """
        compute the logistic function of the input x (some number)
        returns a single number
        """
        return(1/(1+math.exp(-x)))
        "*** YOUR CODE HERE ***"

    def run(self, x):
        """
        Calculates the probability assigned by the logistic regression to a data point x.

        Inputs:
            x: a list with shape (1 x dimensions)
        Returns: a single number (the probability)
        """
        "*** YOUR CODE HERE ***"
        dot = self.DotProduct(self.get_weights(), x)
        return self.sigmoid(dot)
        #print("X === ",x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        if x < 0.5:
            return 0
        else:
            return 1

    def train(self, dataset):
        """
        Train the logistic regression until convergence (this will require at least two loops).
        Use the following internal loop stucture

        for x,y in dataset.iterate_once(1):
            x = nn.as_vector(x)
            y = nn.as_scalar(y)
            ...

        """
        "*** YOUR CODE HERE ***"
        e = 1
        
        while e:
            e = 0

            for x, y in dataset.iterate_once(1):

                x = nn.as_vector(x)
                y = nn.as_scalar(y)
                y_hat = self.run(x)
                
                y = 0 if y == -1 else 1

                loss = -2*y_hat*(y-y_hat)*(1-y_hat)
                q = abs(y-self.get_prediction(y_hat))
                if q == 0:
                    continue
                e += q
                delta_w = np.dot(self.alpha,np.dot(x,loss))
                w = np.subtract(self.w, delta_w)
                
                self.w = w
            print(e)


class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        We did this for you. Nothing for you to do here.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """

        "*** YOUR CODE HERE ***"
        res = nn.DotProduct(self.w,x)
        print(res)

        return res



    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        print(x)
        t = nn.as_scalar(x)

        a = 1
        
        if t < 0.0:
            a = -1

        print(t, '----', a)
        return a
        
    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        e = 1
        
        # while e:
        #     e = 0

        for x, y in dataset.iterate_once(1):

            x = nn.as_vector(x)
            print(x)
            y = nn.as_scalar(y)
            r = self.run(x)
            
            #y = 0 if y == -1 else 1
            print(r)
            # loss = -2*y_hat*(y-y_hat)*(1-y_hat)
            q = abs(y-self.get_prediction(r))
            if q == 0:
                continue
            # e += q
            # delta_w = np.dot(self.alpha,np.dot(x,loss))
            # w = np.subtract(self.w, delta_w)
                
                # self.w = w
class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self, dimensions):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

    def closedFormSolution(self, X, Y):
        """
        Compute the closed form solution for the 2D case
        Input: X,Y are lists
        Output: b0 and b1 where y = b1*x + b0
        """
        "*** YOUR CODE HERE ***"

class PolyRegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self, order):
        # Initialize your model parameters here
        """
        initialize the order of the polynomial, as well as two parameter nodes for weights and bias
        """
        "*** YOUR CODE HERE ***"

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

    def computePolyFeatures(self, point):
        """
        Compute the polynomial features you need from the input x
        NOTE: you will need to unpack the x since it is wrapped in an object
        thus, use the following function call to get the contents of x as a list:
        point_list = nn.as_vector(point)
        Once you do that, create a list of the form (for batch size of n): [[x11, x12, ...], [x21, x22, ...], ..., [xn1, xn2, ...]]
        Once this is done, then use the following code to convert it back into the object
        nn.Constant(nn.list_to_arr(new_point_list))
        Input: a node with shape (batch_size x 1)
        Output: an nn.Constant object with shape (batch_size x n) where n is the number of features generated from point (input)
        """
        "*** YOUR CODE HERE ***"

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

class FashionClassificationModel(object):
    """
    A model for fashion clothing classification using the MNIST dataset.

    Each clothing item is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
