import math
import numpy as np
import autograd.numpy as np
import sys
import warnings
from autograd import grad, elementwise_grad
from random import random, seed
from copy import deepcopy, copy
from typing import Tuple, Callable
from sklearn.utils import resample
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from matplotlib import cm

warnings.simplefilter("error")


class Scheduler:
    """
    Abstract class for Schedulers
    """

    def __init__(self, eta):
        self.eta = eta

    # should be overwritten
    def update_change(self, gradient):
        raise NotImplementedError

    # overwritten if needed
    def reset(self):
        pass


class Constant(Scheduler):
    def __init__(self, eta):
        super().__init__(eta)

    def update_change(self, gradient):
        return self.eta * gradient

    def reset(self):
        pass


class Momentum(Scheduler):
    def __init__(self, eta: float, momentum: float):
        super().__init__(eta)
        self.momentum = momentum
        self.change = 0

    def update_change(self, gradient):
        self.change = self.momentum * self.change + self.eta * gradient
        return self.change

    def reset(self):
        pass


class Adagrad(Scheduler):
    def __init__(self, eta):
        super().__init__(eta)
        self.G_t = None

    def update_change(self, gradient):
        delta = 1e-8  # avoid division by zero

        if self.G_t is None:
            self.G_t = np.zeros((gradient.shape[0], gradient.shape[0]))

        self.G_t += gradient @ gradient.T

        G_t_inverse = 1 / (
            delta + np.sqrt(np.reshape(np.diagonal(self.G_t), (1, self.G_t.shape[0])))
        )
        return self.eta * gradient * G_t_inverse

    def reset(self):
        self.G_t = None


class AdagradMomentum(Scheduler):
    def __init__(self, eta, momentum):
        super().__init__(eta)
        self.G_t = None
        self.momentum = momentum
        self.change = 0

    def update_change(self, gradient):
        delta = 1e-8  # avoid division by zero

        if self.G_t is None:
            self.G_t = np.zeros((gradient.shape[0], gradient.shape[0]))

        self.G_t += gradient @ gradient.T

        G_t_inverse = 1 / (
            delta + np.sqrt(np.reshape(np.diagonal(self.G_t), (self.G_t.shape[0], 1)))
        )
        self.change = self.change * self.momentum + self.eta * gradient * G_t_inverse
        return self.change

    def reset(self):
        self.G_t = None


class RMS_prop(Scheduler):
    def __init__(self, eta, rho):
        super().__init__(eta)
        self.rho = rho
        self.second = 0.0

    def update_change(self, gradient):
        delta = 1e-8  # avoid division by zero
        self.second = self.rho * self.second + (1 - self.rho) * gradient * gradient
        return self.eta * gradient / (np.sqrt(self.second + delta))

    def reset(self):
        self.second = 0.0


class Adam(Scheduler):
    def __init__(self, eta, rho, rho2):
        super().__init__(eta)
        self.rho = rho
        self.rho2 = rho2
        self.moment = 0
        self.second = 0
        self.n_epochs = 1

    def update_change(self, gradient):
        delta = 1e-8  # avoid division by zero

        self.moment = self.rho * self.moment + (1 - self.rho) * gradient
        self.second = self.rho2 * self.second + (1 - self.rho2) * gradient * gradient

        moment_corrected = self.moment / (1 - self.rho**self.n_epochs)
        second_corrected = self.second / (1 - self.rho2**self.n_epochs)

        return self.eta * moment_corrected / (np.sqrt(second_corrected + delta))

    def reset(self):
        self.n_epochs += 1
        self.moment = 0
        self.second = 0


def CostOLS(target):
    def func(predicted):
        return (1 / predicted.shape[0]) * np.sum((target - predicted) ** 2)

    return func


def CostRidge(y, X, w, lmbda):

    def func(X):
        return (1 / X.shape[0]) * np.sum((y - X @ w) ** 2 + ((lmbda / X.shape[0]) * w))

    return func


def CostLogReg(target):

    def func(X):

        return -(1.0 / target.shape[0]) * np.sum(
            (target * np.log(X + 10e-10)) + ((1 - target) * np.log(1 - X + 10e-10))
        )

    return func


def CostCrossEntropy(target):

    def func(X):
        return -(1.0 / target.size) * np.sum(target * np.log(X + 10e-10))

    return func


def identity(X):
    return X


def sigmoid(X):
    try:
        return 1.0 / (1 + np.exp(-X))
    except FloatingPointError:
        return np.where(X > np.zeros(X.shape), np.ones(X.shape), np.zeros(X.shape))


def softmax(X):
    X = X - np.max(X, axis=-1, keepdims=True)
    delta = 10e-10
    return np.exp(X) / (np.sum(np.exp(X), axis=-1, keepdims=True) + delta)


def RELU(X):
    return np.where(X > np.zeros(X.shape), X, np.zeros(X.shape))


def LRELU(X):
    delta = 10e-4
    return np.where(X > np.zeros(X.shape), X, delta * X)


def derivate(func):
    if func.__name__ == "RELU":

        def func(X):
            return np.where(X > 0, 1, 0)

        return func

    elif func.__name__ == "LRELU":

        def func(X):
            delta = 10e-4
            return np.where(X > 0, 1, delta)

        return func

    else:
        return elementwise_grad(func)


class LogisticRegression:
    def __init__(self, epochs=100, batches=10, lam=None):
        self.epochs = epochs
        self.batches = batches
        self.beta_logreg = None
        self.lam = lam

    def GDfit(self, X, y, scheduler=Constant(eta=0.01)):
        """
        Fits a logistic regression model using SGD
        Parameters:
        X: design matrix
        y: target variable
        scheduler: gradient descent method to be used
        """
        n_data, num_features = X.shape
        self.beta_logreg = np.zeros(num_features)
        batch_size = X.shape[0] // self.batches
        for e in range(self.epochs):
            shuffled_indices = np.random.permutation(X.shape[0])
            X = X[shuffled_indices]
            y = y[shuffled_indices]
            for i in range(self.batches):
                if i == self.batches - 1:
                    # If the for loop has reached the last batch, take all that's left
                    X_batch = X[i * batch_size :, :]
                    y_batch = y[i * batch_size :]
                else:
                    X_batch = X[i * batch_size : (i + 1) * batch_size, :]
                    y_batch = y[i * batch_size : (i + 1) * batch_size]
            linear_model = X_batch @ self.beta_logreg
            y_pred = sigmoid(linear_model)
            if not self.lam:
                gradient = (X_batch.T @ (y_pred - y_batch)) / n_data
            else:
                # calculating the gradient from adding L2 regularization to the cost function
                gradient_beta0 = (
                    X_batch.T[0] @ (y_pred - y_batch)
                ) / n_data  # beta_0 is calculated without regularization
                gradient_rest = (
                    X_batch.T[1:] @ (y_pred - y_batch)
                ) / n_data + self.lam / n_data * self.beta_logreg[1:]
                gradient = np.insert(gradient_rest, 0, gradient_beta0, 0)
            change = scheduler.update_change(gradient)
            self.beta_logreg -= change
        return self.beta_logreg

    def predict(self, X):
        linear_model = X @ self.beta_logreg
        y_predicted = sigmoid(linear_model)
        return [1 if i >= 0.5 else 0 for i in y_predicted]


class FFNN:
    """
    Description:
    ------------
        Feed Forward Neural Network with interface enabling flexible design of a
        nerual networks architecture and the specification of activation function
        in the hidden layers and output layer respectively. This model can be used
        for both regression and classification problems, depending on the output function.

    Attributes:
    ------------
        I   dimensions (tuple[int]): A list of positive integers, which specifies the
            number of nodes in each of the networks layers. The first integer in the array
            defines the number of nodes in the input layer, the second integer defines number
            of nodes in the first hidden layer and so on until the last number, which
            specifies the number of nodes in the output layer.
        II  hidden_func (Callable): The activation function for the hidden layers
        III output_func (Callable): The activation function for the output layer
        IV  cost_func (Callable): Our cost function
        V   seed (int): Sets random seed, makes results reproducible
    """

    def __init__(
        self,
        dimensions: tuple[int],
        hidden_func: Callable = sigmoid,
        output_func: Callable = lambda x: x,
        cost_func: Callable = CostOLS,
        seed: int = None,
    ):
        self.dimensions = dimensions
        self.hidden_func = hidden_func
        self.output_func = output_func
        self.cost_func = cost_func
        self.seed = seed
        self.weights = list()
        self.schedulers_weight = list()
        self.schedulers_bias = list()
        self.a_matrices = list()
        self.z_matrices = list()
        self.classification = None

        self.reset_weights()
        self._set_classification()

    def fit(
        self,
        X: np.ndarray,
        t: np.ndarray,
        scheduler: Scheduler,
        batches: int = 1,
        epochs: int = 100,
        lam: float = 0,
        X_val: np.ndarray = None,
        t_val: np.ndarray = None,
        tol: float = 0.0001,
        show_progress_bar: bool = True,
        show_tol_reached: bool = True,
        show_scheduler_info: bool = True,
        n_iter_no_change: int = 1,
    ):
        """
        Description:
        ------------
            This function performs the training the neural network by performing the feedforward and backpropagation
            algorithm to update the networks weights.

        Parameters:
        ------------
            I    X (np.ndarray) : training data
            II   t (np.ndarray) : target data
            III  scheduler (Scheduler) : specified scheduler (algorithm for optimization of gradient descent)
            IV   scheduler_args (list[int]) : list of all arguments necessary for scheduler

        Optional Parameters:
        ------------
            V    batches (int) : number of batches the datasets are split into, default equal to 1
            VI   epochs (int) : number of iterations used to train the network, default equal to 100
            VII  lam (float) : regularization hyperparameter lambda
            VIII X_val (np.ndarray) : validation set
            IX   t_val (np.ndarray) : validation target set
            X    tol (float) : tolerance for continued training
            XI   show_progress_bar (bool) : print progress bar during training
            XII  show_tol_reached (bool) : print the epoch at which training stops by hitting tolerance
            XIII show_scheduler_info (bool) : print information about scehduler
            XIV  n_iter_no_change (int) : number of consecutive epochs with a training error below tol needed to stop training

        Returns:
        ------------
            I   scores (dict) : A dictionary containing the performance metrics of the model.
                The number of the metrics depends on the parameters passed to the fit-function.

        """

        # setup
        if self.seed is not None:
            np.random.seed(self.seed)

        val_set = False
        if X_val is not None and t_val is not None:
            val_set = True

        # creating arrays for score metrics
        train_errors = np.empty(epochs)
        train_errors.fill(np.nan)
        val_errors = np.empty(epochs)
        val_errors.fill(np.nan)
        R2_scores = np.empty(epochs)

        train_accs = np.empty(epochs)
        train_accs.fill(np.nan)
        val_accs = np.empty(epochs)
        val_accs.fill(np.nan)

        self.schedulers_weight = list()
        self.schedulers_bias = list()

        batch_size = X.shape[0] // batches

        X, t = resample(X, t)  # purpose? Supposed to shuffle data? replace=False?

        # this function returns a function valued only at X
        cost_function_train = self.cost_func(t)
        if val_set:
            cost_function_val = self.cost_func(t_val)

        # create schedulers for each weight matrix
        for i in range(len(self.weights)):
            self.schedulers_weight.append(copy(scheduler))
            self.schedulers_bias.append(copy(scheduler))

        if show_scheduler_info:
            print(f"{scheduler.__class__.__name__}: Eta={scheduler.eta}, Lambda={lam}")

        try:
            for e in range(epochs):
                for i in range(batches):  # Should data be shuffled?
                    # allows for minibatch gradient descent
                    if i == batches - 1:
                        # If the for loop has reached the last batch, take all thats left
                        X_batch = X[i * batch_size :, :]
                        t_batch = t[i * batch_size :, :]
                    else:
                        X_batch = X[i * batch_size : (i + 1) * batch_size, :]
                        t_batch = t[i * batch_size : (i + 1) * batch_size, :]

                    self._feedforward(X_batch)
                    self._backpropagate(X_batch, t_batch, lam)

                # reset schedulers for each epoch (some schedulers pass in this call)
                for scheduler in self.schedulers_weight:
                    scheduler.reset()

                for scheduler in self.schedulers_bias:
                    scheduler.reset()

                # computing performance metrics
                pred_train = self.predict(X)
                train_error = cost_function_train(pred_train)

                train_errors[e] = train_error

                if val_set:
                    pred_val = self.predict(X_val)
                    val_error = cost_function_val(pred_val)
                    val_errors[e] = val_error
                    if self.cost_func.__name__ == "CostOLS":
                        R2 = 1 - np.sum((t_val - pred_val) ** 2) / np.sum(
                            (t_val - np.mean(t_val)) ** 2
                        )
                        R2_scores[e] = R2

                if self.classification:
                    train_acc = self._accuracy(self.predict(X), t)
                    train_accs[e] = train_acc
                    if val_set:
                        val_acc = self._accuracy(pred_val, t_val)
                        val_accs[e] = val_acc

                if show_progress_bar:
                    # printing progress bar
                    progression = e / epochs
                    print_length = self._progress_bar(
                        progression,
                        train_error=train_errors[e],
                        train_acc=train_accs[e],
                        val_error=val_errors[e],
                        val_acc=val_accs[e],
                    )
                # tolerance check
                if e > 0:
                    tols = np.array([tol for _ in range(n_iter_no_change)])
                    prev_abs_errors = np.abs(
                        [
                            train_errors[i] - train_errors[i - 1]
                            for i in range(e, e - n_iter_no_change, -1)
                        ]
                    )
                    comparison = prev_abs_errors < tols
                    if False not in comparison:
                        if show_tol_reached:
                            print(
                                f"Training cancelled at epoch = {e} due to tolerance reached"
                            )
                        break
        except KeyboardInterrupt:
            # allows for stopping training at any point and seeing the result
            pass

        if show_progress_bar:
            # visualization of training progression (similiar to tensorflow progression bar)
            sys.stdout.write("\r" + " " * print_length)
            sys.stdout.flush()
            self._progress_bar(
                1,
                train_error=train_errors[e],
                train_acc=train_accs[e],
                val_error=val_errors[e],
                val_acc=val_accs[e],
            )
            sys.stdout.write("")

        # return performance metrics for the entire run
        scores = dict()

        scores["train_errors"] = train_errors

        if val_set:
            scores["val_errors"] = val_errors
            if self.cost_func.__name__ == "CostOLS":
                scores["R2_scores"] = R2_scores

        if self.classification:
            scores["train_accs"] = train_accs

            if val_set:
                scores["val_accs"] = val_accs

        return scores

    def predict(self, X: np.ndarray, *, threshold=0.5):
        """
         Description:
         ------------
             Performs prediction after training of the network has been finished.

         Parameters:
        ------------
             I   X (np.ndarray): The design matrix, with n rows of p features each

         Optional Parameters:
         ------------
             II  threshold (float) : sets minimal value for a prediction to be predicted as the positive class
                 in classification problems

         Returns:
         ------------
             I   z (np.ndarray): A prediction vector (row) for each row in our design matrix
                 This vector is thresholded if regression=False, meaning that classification results
                 in a vector of 1s and 0s, while regressions in an array of decimal numbers

        """

        predict = self._feedforward(X)

        if self.classification:
            return np.where(predict > threshold, 1, 0)
        else:
            return predict

    def reset_weights(self):
        """
        Description:
        ------------
            Resets/Reinitializes the weights in order to train the network for a new problem.

        """
        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = list()
        for i in range(len(self.dimensions) - 1):
            weight_array = np.random.randn(
                self.dimensions[i] + 1, self.dimensions[i + 1]
            )
            weight_array[0, :] = np.random.randn(self.dimensions[i + 1]) * 0.01

            self.weights.append(weight_array)

    def _feedforward(self, X: np.ndarray):
        """
        Description:
        ------------
            Calculates the activation of each layer starting at the input and ending at the output.
            Each following activation is calculated from a weighted sum of each of the preceeding
            activations (except in the case of the input layer).

        Parameters:
        ------------
            I   X (np.ndarray): The design matrix, with n rows of p features each

        Returns:
        ------------
            I   z (np.ndarray): A prediction vector (row) for each row in our design matrix
        """

        # reset matrices
        self.a_matrices = list()
        self.z_matrices = list()

        # if X is just a vector, make it into a matrix
        if len(X.shape) == 1:
            X = X.reshape((1, X.shape[0]))

        # X = n x p

        # Add a coloumn of zeros as the first coloumn of the design matrix, in order        # ones?
        # to add bias to our data
        bias = np.ones((X.shape[0], 1)) * 0.01  # Why so small?
        X = np.hstack([bias, X])

        # X = n x (p + 1)

        # a^0, the nodes in the input layer (one a^0 for each row in X - where the
        # exponent indicates layer number).
        a = X  # n x (p+1)
        self.a_matrices.append(a)
        self.z_matrices.append(a)

        # The feed forward algorithm
        for i in range(len(self.weights)):
            if i < len(self.weights) - 1:
                z = a @ self.weights[i]  # first: (n x (p+1)) x ((p+1) x n_2) = n x n_2
                self.z_matrices.append(z)
                a = self.hidden_func(z)
                # bias column again added to the data here
                bias = np.ones((a.shape[0], 1)) * 0.01
                a = np.hstack([bias, a])  # n x (n_2 + 1)
                self.a_matrices.append(a)
            else:
                try:
                    # a^L, the nodes in our output layers
                    z = a @ self.weights[i]
                    a = self.output_func(z)
                    self.a_matrices.append(a)
                    self.z_matrices.append(z)
                except Exception as OverflowError:
                    print(
                        "OverflowError in fit() in FFNN\nHOW TO DEBUG ERROR: Consider lowering your learning rate or scheduler specific parameters such as momentum, or check if your input values need scaling"
                    )

        # this will be a^L
        return a

    def _backpropagate(self, X, t, lam):
        """
        Description:
        ------------
            Performs the backpropagation algorithm. In other words, this method
            calculates the gradient of all the layers starting at the
            output layer, and moving from right to left accumulates the gradient until
            the input layer is reached. Each layers respective weights are updated while
            the algorithm propagates backwards from the output layer (auto-differentation in reverse mode).

        Parameters:
        ------------
            I   X (np.ndarray): The design matrix, with n rows of p features each.
            II  t (np.ndarray): The target vector, with n rows of p targets.
            III lam (float32): regularization parameter used to punish the weights in case of overfitting

        Returns:
        ------------
            No return value.

        """
        out_derivative = derivate(self.output_func)
        hidden_derivative = derivate(self.hidden_func)

        for i in range(len(self.weights) - 1, -1, -1):
            # delta terms for output
            if i == len(self.weights) - 1:
                # for multi-class classification
                if self.output_func.__name__ == "softmax":
                    delta_matrix = self.a_matrices[i + 1] - t
                # for single class classification
                else:
                    cost_func_derivative = grad(self.cost_func(t))
                    delta_matrix = out_derivative(
                        self.z_matrices[i + 1]  # n x n_out
                    ) * cost_func_derivative(self.a_matrices[i + 1])

            # delta terms for hidden layer
            else:
                delta_matrix = (
                    self.weights[i + 1][1:, :] @ delta_matrix.T
                ).T * hidden_derivative(self.z_matrices[i + 1])

            # calculate gradient
            gradient_weights = (
                self.a_matrices[i][:, 1:].T @ delta_matrix
            )  # n_i x n @ n x n_i+1
            gradient_bias = np.sum(delta_matrix, axis=0).reshape(
                1, delta_matrix.shape[1]  # 1 x n_i+1
            )

            # regularization term
            gradient_weights += self.weights[i][1:, :] * lam

            # use scheduler
            update_matrix = np.vstack(
                [
                    self.schedulers_bias[i].update_change(gradient_bias),
                    self.schedulers_weight[i].update_change(gradient_weights),
                ]
            )

            # update weights and bias
            self.weights[i] -= update_matrix

    def _accuracy(self, prediction: np.ndarray, target: np.ndarray):
        """
        Description:
        ------------
            Calculates accuracy of given prediction to target

        Parameters:
        ------------
            I   prediction (np.ndarray): vector of predicitons output network
                (1s and 0s in case of classification, and real numbers in case of regression)
            II  target (np.ndarray): vector of true values (What the network ideally should predict)

        Returns:
        ------------
            A floating point number representing the percentage of correctly classified instances.
        """
        assert prediction.size == target.size
        return np.average((target == prediction))

    def _set_classification(self):
        """
        Description:
        ------------
            Decides if FFNN acts as classifier (True) og regressor (False),
            sets self.classification during init()
        """
        self.classification = False
        if (
            self.cost_func.__name__ == "CostLogReg"
            or self.cost_func.__name__ == "CostCrossEntropy"
        ):
            self.classification = True

    def _progress_bar(self, progression, **kwargs):
        """
        Description:
        ------------
            Displays progress of training
        """
        print_length = 40
        num_equals = int(progression * print_length)
        num_not = print_length - num_equals
        arrow = ">" if num_equals > 0 else ""
        bar = "[" + "=" * (num_equals - 1) + arrow + "-" * num_not + "]"
        perc_print = self._format(progression * 100, decimals=5)
        line = f"  {bar} {perc_print}% "

        for key in kwargs:
            if not np.isnan(kwargs[key]):
                value = self._format(kwargs[key], decimals=4)
                line += f"| {key}: {value} "
        sys.stdout.write("\r" + line)
        sys.stdout.flush()
        return len(line)

    def _format(self, value, decimals=4):
        """
        Description:
        ------------
            Formats decimal numbers for progress bar
        """
        if value > 0:
            v = value
        elif value < 0:
            v = -10 * value
        else:
            v = 1
        n = 1 + math.floor(math.log10(v))
        if n >= decimals - 1:
            return str(round(value))
        return f"{value:.{decimals-n-1}f}"


def simple_function(coef, x):
    a_0, a_1, a_2 = coef
    return a_0 + a_1 * x + a_2 * x**2


def FrankeFunction(x, y):
    "Franke function definition"
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-((9 * x - 7) ** 2) / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4


def SGD(
    X,
    y,
    n_epochs,
    gradient_fn,
    scheduler,
    tol,
    cost_function,
    lmbda=None,
    batch_size=None,
):
    """
    Performs Stochastic Gradient Descent (SGD) with mini-batches.

    Parameters:
    X : ndarray, shape (m, n), design matrix
    y : ndarray, shape (m, ), target variable
    n_epochs : int, number of epochs
    batch_size : int, size of each mini-batch
    gradient_fn : function, function to compute the gradients
    scheduler : object, gradient descent method to be used
    tol : float, convergence tolerance
    cost_function : function, (CostOLS or CostRidge)
    lmbda : float, regularization parameter for Ridge (only used for CostRidge)

    Returns:
    w : ndarray, shape (n, ), updated weights after n_epochs
    costs : list, cost value at each epoch
    """
    m, n = X.shape
    w = np.zeros(n)
    costs = []
    cost_prev = None

    # Set default batch_size if not provided
    if batch_size is None:
        batch_size = m  # full batch gradient descent

    for epoch in range(n_epochs):
        shuffled_indices = np.random.permutation(m)
        X_shuffled = X[shuffled_indices]
        y_shuffled = y[shuffled_indices]

        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i : i + batch_size]
            y_batch = y_shuffled[i : i + batch_size]
            if lmbda:
                gradients = gradient_fn(X_batch, y_batch, w, lmbda)  # Ridge
            else:
                gradients = gradient_fn(X_batch, y_batch, w)  # OLS

            update = scheduler.update_change(gradients)

            # Ensure same shape, for instance Adagrad class reshapes into column vector
            if update.shape != w.shape:
                if len(update.shape) > 1:
                    update = np.diag(update)  # Handle matrix case
                else:
                    update = update.flatten()  # Handle row vector case

            w -= update

        # Cost for full dataset
        if lmbda:
            cost = cost_function(y, X, w, lmbda)  # Ridge
        else:
            cost = cost_function(y, X, w)  # OLS

        costs.append(cost)

        # Check for convergence
        if cost_prev is not None and abs(cost_prev - cost) < tol:
            print(f"Converged after {epoch+1} epochs")
            break

        cost_prev = cost

        if len(costs) == n_epochs:
            print(f"Did not converge for {n_epochs} epochs")

    return w, costs, epoch + 1  # Return weights, costs, and number of iterations


def get_MSE_R2_OLS_Ridge(X, z, p, model, lambda_=0, return_train_MSE=False):
    """
    Perform regression with a polynomial transformation of degree p, and return the MSE and R^2 calculated on a test set.
    The 'model' parameter dictates which regression method is used ('OLS' or 'ridge').
    """
    poly = PolynomialFeatures(degree=p)
    X_poly = poly.fit_transform(X)

    # Scale and determine test/train split
    # Scale data according to model
    if model == "ridge":
        X_poly = X_poly[:, 1:]  # Removing first column of all ones
        X_poly = X_poly - np.mean(X_poly, axis=0)  # Centering
        X_poly = X_poly / np.std(X_poly, axis=0)  # Scaling
    else:  # OLS model: center and scale only non-intercept columns
        X_poly[:, 1:] = X_poly[:, 1:] - np.mean(X_poly[:, 1:], axis=0)
        X_poly[:, 1:] = X_poly[:, 1:] / np.std(X_poly[:, 1:], axis=0)

    X_train, X_test, z_train, z_test = train_test_split(
        X_poly, z, test_size=0.2, random_state=42
    )

    I = np.eye(X_train.shape[1])  # Identity matrix
    beta_hat = []
    z_fitted = []
    MSE_train = 0

    if model == "OLS":
        X_T_X_inv = np.linalg.solve(
            X_train.T @ X_train, I
        )  # Using np.linalg.solve for stability
        beta_hat = X_T_X_inv @ (X_train.T @ z_train)
        z_fitted = X_test @ beta_hat
        z_fitted_train = X_train @ beta_hat
        MSE_train = np.mean((z_train - z_fitted_train) ** 2)

    elif model == "ridge":
        ridge_inv = np.linalg.solve(X_train.T @ X_train + lambda_ * I, I)
        beta_hat_ridge = ridge_inv @ (X_train.T @ z_train)
        beta_0 = np.mean(z_train)
        z_fitted = beta_0 + X_test @ beta_hat_ridge
        z_fitted_train = beta_0 + X_train @ beta_hat_ridge
        beta_hat = np.concatenate(([beta_0], beta_hat_ridge))
        MSE_train = np.mean((z_train - z_fitted_train) ** 2)

    MSE = np.mean((z_test - z_fitted) ** 2)
    R2 = 1 - np.sum((z_test - z_fitted) ** 2) / np.sum((z_test - np.mean(z_test)) ** 2)

    if return_train_MSE:
        return MSE, R2, MSE_train
    return MSE, R2


def get_MSE_R2_beta(X, z, p, model, lambda_=0, return_train_MSE=False):
    """
    Only to be used by plot_polynom function
    """
    poly = PolynomialFeatures(degree=p)
    X_poly = poly.fit_transform(X)

    # Scale and determine test/train split
    # Scale data according to model
    if model == "ridge" or model == "lasso":
        X_poly = X_poly[:, 1:]  # Removing first collumn of all ones
        X_poly = X_poly - np.mean(X_poly, axis=0)  # centering
        X_poly = X_poly / np.std(X_poly, axis=0)  # scaling
    else:  # NB: will cause trouble for column of 1s
        X_poly[:, 1:] = X_poly[:, 1:] - np.mean(
            X_poly[:, 1:], axis=0
        )  # centering, keeping the 1 column
        X_poly[:, 1:] = X_poly[:, 1:] / np.std(
            X_poly[:, 1:], axis=0
        )  # scaling, still keeping the 1 column
    X_train, X_test, z_train, z_test = train_test_split(X_poly, z)

    I = np.eye(X_train.shape[1])  # identity matrix
    beta_hat = []
    z_fitted = []
    MSE_train = 0
    if model == "OLS":
        X_T_X_inv = np.linalg.solve(
            X_train.T @ X_train, I
        )  # using np.linalg.solve for more stable inversion
        beta_hat = X_T_X_inv @ (X_train.T @ z_train)
        z_fitted = X_test @ beta_hat
        z_fitted_train = X_train @ beta_hat
        MSE_train = np.mean((z_train - z_fitted_train) ** 2)

    elif model == "ridge":
        ridge_inv = np.linalg.solve(X_train.T @ X_train + lambda_ * I, I)
        beta_hat_ridge = ridge_inv @ (X_train.T @ z_train)
        beta_0 = np.mean(z_train)
        z_fitted = beta_0 + X_test @ beta_hat_ridge
        z_fitted_train = beta_0 + X_train @ beta_hat_ridge
        beta_hat = np.concatenate(([beta_0], beta_hat_ridge))
        MSE_train = np.mean((z_train - z_fitted_train) ** 2)

    MSE = np.mean((z_test - z_fitted) ** 2)
    R2 = 1 - np.sum((z_test - z_fitted) ** 2) / np.sum((z_test - np.mean(z_test)) ** 2)
    if return_train_MSE:
        return MSE, R2, beta_hat, poly, MSE_train
    return MSE, R2, beta_hat, poly


def plot_polynom(X, x_mesh, y_mesh, z, degrees, z_NN, model="OLS"):
    """Performs OLS on the data (X, z), and plots the resulting polynomial over the meshgrid (x_mesh, y_mesh) for each degree in degrees"""
    fig = plt.figure(figsize=(18, 12))
    # Plotting Franke
    ax_true = fig.add_subplot(1, len(degrees) + 2, 1, projection="3d")
    z_true = FrankeFunction(x_mesh, y_mesh)
    ax_true.plot_surface(
        x_mesh, y_mesh, z_true, cmap=cm.coolwarm, linewidth=0, antialiased=False
    )

    ax_true.set_title("True Franke function")
    ax_true.set_zlim(-0.10, 1.40)
    ax_true.zaxis.set_major_locator(LinearLocator(10))
    ax_true.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

    # Iterating over each degree
    for idx, degree in enumerate(degrees):
        sub_plot = fig.add_subplot(1, len(degrees) + 2, idx + 2, projection="3d")

        MSE, _, beta_hat, poly = get_MSE_R2_beta(X, z, p=degree, model=model)

        X_grid_poly = poly.transform(X)
        z_pred = np.dot(X_grid_poly, beta_hat)
        z_pred = z_pred.reshape(x_mesh.shape)

        # Plotting the surfaces for the different polynomials
        sub_plot.plot_surface(
            x_mesh, y_mesh, z_pred, cmap=cm.coolwarm, linewidth=0, antialiased=False
        )

        # Customize the z axis
        sub_plot.set_zlim(-0.10, 1.40)
        sub_plot.zaxis.set_major_locator(LinearLocator(10))
        sub_plot.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
        sub_plot.set_title(f"OLS, Degree: {degree}")

    ax_NN = fig.add_subplot(1, len(degrees) + 2, len(degrees) + 2, projection="3d")
    ax_NN.plot_surface(
        x_mesh, y_mesh, z_NN, cmap=cm.coolwarm, linewidth=0, antialiased=False
    )

    ax_NN.set_title("Optimal FFNN surface")
    ax_NN.set_zlim(-0.10, 1.40)
    ax_NN.zaxis.set_major_locator(LinearLocator(10))
    ax_NN.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

    plt.savefig("../Figures/plot_polynom.pdf", bbox_inches="tight")
    plt.show()
