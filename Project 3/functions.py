import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class DiffusionNN(tf.keras.Model):
    """
    FFNN model using tf.keras.Model with Dense layers and tunable network architecture
    """

    def __init__(self, num_hidden_layers, num_nodes_per_layer, activation):
        super(DiffusionNN, self).__init__()
        self.hidden_layers = [
            tf.keras.layers.Dense(num_nodes_per_layer, activation=activation)
            for _ in range(num_hidden_layers)
        ]
        self.output_layer = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)


def trial_solution(x, t, nn_output):
    """
    Trial solution satisfying the initial conditions of the PDE
    """
    return (1 - t) * tf.sin(np.pi * x) + x * (1 - x) * t * nn_output


def compute_loss(model, x, t):
    """
    Loss function used in FFNN model. Defined as the mean square difference of the partial derivative of the trial solution w.r.t. t and the second derivative w.r.t. x
    """
    with tf.GradientTape(persistent=True) as tape1:
        tape1.watch([x, t])
        nn_output = model(tf.concat([x, t], axis=1))
        u_t = trial_solution(x, t, nn_output)
        u_t_t = tape1.gradient(u_t, t)
        u_x = tape1.gradient(u_t, x)
    u_xx = tape1.gradient(u_x, x)
    del tape1

    pde_residual = u_t_t - u_xx
    loss = tf.reduce_mean(tf.square(pde_residual))
    return loss


def train_model(model, optimizer, x, t, epochs, return_loss=False):
    """
    Train FFNN model with a given optimizer and number of epochs
    """
    if return_loss:
        loss_array = np.zeros(epochs)
        U_analytical = np.exp(-np.pi**2 * t) * np.sin(np.pi * x)
        # u_true = tf.reshape(U_analytical, (-1, 1))
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss = compute_loss(model, x, t)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if return_loss:
            loss_compare = np.mean(
                (trial_solution(x, t, model(tf.concat([x, t], axis=1))) - U_analytical)
                ** 2
            )
            loss_array[epoch] = loss_compare
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.numpy()}")

    if return_loss:
        return loss_array


def analytical_solution(x, t):
    """
    Analytical solution of diffusion equation given our initial conditions
    """
    return np.exp(-np.pi**2 * t) * np.sin(np.pi * x)


def reshape_for_rnn(x, t):
    """
    Reshape data for RNN input
    """
    return tf.reshape(tf.concat([x, t], axis=1), (-1, 1, 2))


def loss_fn(model, x, t, u_true):
    """
    Loss function used in RNN model. Defined as the MSE between the predicted values and the analytical solution
    """
    inputs = reshape_for_rnn(x, t)
    nn_output = model(inputs)
    u_pred = trial_solution(x, t, nn_output)
    return tf.reduce_mean(tf.square(u_pred - u_true))


def build_rnn(num_hidden_layers, num_units_per_layer, activation):
    """
    Build a RNN using tensorflow/keras with tunable hyperparameters
    """
    layers = [tf.keras.layers.InputLayer(input_shape=(1, 2))]
    for _ in range(num_hidden_layers):
        layers.append(
            tf.keras.layers.SimpleRNN(
                num_units_per_layer, activation=activation, return_sequences=True
            )
        )
    layers.append(
        tf.keras.layers.SimpleRNN(
            num_units_per_layer, activation=activation, return_sequences=False
        )
    )
    layers.append(tf.keras.layers.Dense(1))
    return tf.keras.Sequential(layers)


def train_model_RNN(model, x, t, u_true, optimizer, epochs, return_loss=False):
    """
    Train RNN model with given optimizer and number of epochs
    """
    loss_array = np.zeros(epochs)
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss = loss_fn(model, x, t, u_true)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        loss_array[epoch] = loss
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.numpy()}")
    if return_loss:
        return loss_array
    return model


def plot_solution(X, T, U, title, ax, cmap="coolwarm"):
    """
    Method for plotting solution surface
    """
    ax.plot_surface(X, T, U, cmap=cmap)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("u(x, t)")
    ax.set_title(title)


def initial(x):
    """Initial condition for diffusion equation"""
    return np.sin(np.pi * x)


def explicit_scheme(n: int, dt: int) -> np.ndarray:
    """For a given number of steps in space, n, and in time, dt,
    compute the explicit scheme solution to a grid of size (n+1)x(dt+1)"""

    # Assess stability criterion
    alpha = n**2 / dt
    print(f"alpha = {alpha}")

    dx = 1.0 / n

    # Initialize the vectors
    u = np.zeros((n + 1, dt + 1))

    # Apply initial conditions for t=0
    for i in range(1, n):
        x = i * dx
        u[i, 0] = initial(x)

    # Time integration
    for t in range(1, dt + 1):
        for i in range(1, n):
            # Discretized differential equation
            u[i, t] = (
                alpha * u[i - 1, t - 1]
                + (1 - 2 * alpha) * u[i, t - 1]
                + alpha * u[i + 1, t - 1]
            )

    return u


def analytical_solution(x, t):
    """
    Analytical solution of diffusion equation given our initial conditions
    """
    return np.exp(-np.pi**2 * t) * np.sin(np.pi * x)
