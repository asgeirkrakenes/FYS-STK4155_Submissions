
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error


def FrankeFunction(x, y):
    "Franke function definition"
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-((9 * x - 7) ** 2) / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4


def test_train_split(X, z, train_size=0.8):
    """Split data into training and test set"""
    indices = np.random.permutation(len(z))
    split_point = int(train_size * len(z))
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]
    X_train = X[train_indices]
    X_test = X[test_indices]
    z_train = z[train_indices]
    z_test = z[test_indices]
    return X_train, X_test, z_train, z_test


def get_MSE_R2_beta(X, z, p, model, lambda_ = 0, return_train_MSE = False):
    """Perform regression with a polynomial transformation of degree p, and return the MSE and R^2 calculated on a test set.
    The 'model' parameter dictates which regression methos is use (OLS, ridge or lasso)
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
    X_train, X_test, z_train, z_test = test_train_split(X_poly, z)

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

    elif model == "lasso":
        lasso = linear_model.Lasso(alpha=lambda_, fit_intercept=True)
        lasso = lasso.fit(X_train, z_train)
        z_fitted = lasso.predict(X_test)
        z_fitted_train = lasso.predict(X_train)
        beta_hat = np.concatenate(([lasso.intercept_], lasso.coef_))
        MSE_train = np.mean((z_train - z_fitted_train) ** 2)

    MSE = np.mean((z_test - z_fitted) ** 2)
    R2 = 1 - np.sum((z_test - z_fitted) ** 2) / np.sum((z_test - np.mean(z_test)) ** 2)
    if return_train_MSE:
        return MSE, R2, beta_hat, poly, MSE_train
    return MSE, R2, beta_hat, poly


def plot_polynom(X, x_mesh, y_mesh, z, degrees, model="OLS"):
    """Performs OLS on the data (X, z), and plots the resulting polynomial over the meshgrid (x_mesh, y_mesh) for each degree in degrees"""
    fig = plt.figure(figsize=(18, 12))
    # Plotting Franke
    ax_true = fig.add_subplot(1, len(degrees) + 1, 1, projection="3d")
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
        sub_plot = fig.add_subplot(1, len(degrees) + 1, idx + 2, projection="3d")

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
        sub_plot.set_title(f"Degree: {degree}\n MSE: {MSE:.4f}")

    plt.savefig("plot_polynom.pdf", bbox_inches="tight")
    plt.show()


# Needs to be made compatible with the rest of the functions. Also add docstring.
def bootstrap_ols(X, z, num_bootstrap, degree):
    predictions = []
    mse_train_scores = []
    mse_test_scores = []

    # Split data into training and test sets
    X_train, X_test, z_train, z_test = test_train_split(X, z, train_size = 0.8)

    # Create polynomial features
    poly = PolynomialFeatures(degree = degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    for _ in range(num_bootstrap):
        # Resample the training data with replacement
        X_resampled, z_resampled = resample(X_train_poly, z_train)
        
        # Fit the polynomial regression model on the bootstrapped sample
        model = linear_model.LinearRegression()
        model.fit(X_resampled, z_resampled)
        
        # Predictions
        z_train_pred = model.predict(X_resampled)
        z_test_pred = model.predict(X_test_poly)
        
        # Compute MSE for training and test sets
        mse_train_scores.append(mean_squared_error(z_resampled, z_train_pred))
        mse_test_scores.append(mean_squared_error(z_test, z_test_pred))

        # Collect predictions for bias-variance computation
        predictions.append(z_test_pred)

    # Calculate the mean MSE over all bootstrap samples
    mean_mse_train = np.mean(mse_train_scores)
    mean_mse_test = np.mean(mse_test_scores)
    
    # Convert the list of predictions to a numpy array
    predictions = np.array(predictions)
    
    # Calculate the bias
    bias_squared = np.mean((np.mean(predictions, axis=0) - z_test)**2)
    
    # Calculate variance
    variance = np.mean(np.var(predictions, axis=0))
    
    return mean_mse_train, mean_mse_test, bias_squared, variance

def cv_get_MSE_R2_beta(X_train, X_test, z_train, z_test, model, lambda_=0):
    """Perform regression with a polynomial transformation of degree p, and return the MSE and R^2 calculated on a test set.
    The 'model' parameter dictates which regression methos is use (OLS, ridge or lasso)
    """
    I = np.eye(X_train.shape[1])  # identity matrix
    beta_hat = []
    z_fitted = []
    if model == "OLS":
        X_T_X_inv = np.linalg.solve(
            X_train.T @ X_train, I
        )  # using np.linalg.solve for more stable inversion
        beta_hat = X_T_X_inv @ (X_train.T @ z_train)
        z_fitted = X_test @ beta_hat

    elif model == "ridge":
        ridge_inv = np.linalg.solve(X_train.T @ X_train + lambda_ * I, I)
        beta_hat_ridge = ridge_inv @ (X_train.T @ z_train)
        beta_0 = np.mean(z_train)
        z_fitted = beta_0 + X_test @ beta_hat_ridge
        beta_hat = np.concatenate(([beta_0], beta_hat_ridge))

    elif model == "lasso":
        lasso = linear_model.Lasso(alpha=lambda_, fit_intercept=True)
        z_fitted = lasso.fit(X_train, z_train).predict(X_test)
        beta_hat = np.concatenate(([lasso.intercept_], lasso.coef_))

    MSE = np.mean((z_test - z_fitted) ** 2)
    R2 = 1 - np.sum((z_test - z_fitted) ** 2) / np.sum((z_test - np.mean(z_test)) ** 2)
    return MSE, R2, beta_hat


def k_fold_cv(X, z, model, k, p=1, lambda_=0):
    """
    k-fold cross-validation for a given model with following parameters:

    X: feature matrix, i.e. coordinates [x,y]
    z: target vector, i.e. altitudes
    p: polynomial degree
    lambda_: penalization parameter for ridge and lasso regression


    """
    # Create fold indices
    kf = KFold(n_splits=k, shuffle=True)
    # mse_train_scores = []
    mse_test_scores = []

    # Create design matrix by polynomial transformation of feature matrix X
    poly = PolynomialFeatures(degree=p)
    X_poly = poly.fit_transform(X)

    # Scale polynomial design matrix according to model
    if model == "ridge" or model == "lasso":
        X_poly = X_poly[:, 1:]  # Removing first collumn of all ones
        X_poly = X_poly - np.mean(X_poly, axis=0)  # centering
        X_poly = X_poly / np.std(X_poly, axis=0)  # scaling
    else:
        X_poly[:, 1:] = X_poly[:, 1:] - np.mean(
            X_poly[:, 1:], axis=0
        )  # centering, keeping the 1 column
        X_poly[:, 1:] = X_poly[:, 1:] / np.std(
            X_poly[:, 1:], axis=0
        )  # scaling, still keeping the 1 column

    # Train and test across data folds
    for train_index, test_index in kf.split(X_poly):
        # Train/test split according to fold indices
        X_train, X_test = X_poly[train_index], X_poly[test_index]
        z_train, z_test = z[train_index], z[test_index]

        # Train model
        MSE_test, R2, beta_hat = cv_get_MSE_R2_beta(
            X_train, X_test, z_train, z_test, model=model, lambda_=lambda_
        )
        mse_test = MSE_test
        mse_test_scores.append(mse_test)

    return np.mean(mse_test_scores)
