import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt

# Generate ground truth polynomial.
def gen_ground_truth_model():
    rng = np.random.default_rng()
    features = rng.integers(2, 10)
    curve = rng.uniform(-10, 10, (features, 1))
    return curve

# Generate data.
def gen_data(w_gt, n=100, stdev=1):
    rng = np.random.default_rng()
    x = rng.uniform(-1.5, 1.5, (n, 1))
    x = np.concatenate((x, np.ones((n, 1))), axis=1)
    for i in range(2, w_gt.shape[0]):
        x = np.concatenate((np.array([x[:, -2]]).T ** i, x), axis=1)

    # Generate label.
    y = x @ w_gt
    
    # Add element-wise gaussian noise to each label.
    y += rng.normal(0, stdev, (n, 1))

    return x, y

# Find the closed form solution for w.
def closed_form(x, y, reg):
    w = npla.inv(x.T@x+reg*np.eye(x.shape[1]))@x.T@y
    return w

# Split the data into training and testing sets.
def split_data(x, y, k, splits):
    n = x.shape[0]
    low = k*n//splits
    high = (k+1)*n//splits
    x_train = np.concatenate((x[:low,:], x[high:,:]), axis=0)
    y_train = np.concatenate((y[:low,:], y[high:,:]), axis=0)
    x_test = x[low:high,:]
    y_test = y[low:high,:]
    return x_train, y_train, x_test, y_test

# perform cross validation.
def cross_validation(x, y, reg, splits=10):
    cumul_loss = 0
    for k in range(splits):
        x_train, y_train, x_test, y_test = split_data(x, y, k, splits)
        w = closed_form(x_train, y_train, reg)
        loss = 1/2*npla.norm(x_test@w - y_test)**2
        cumul_loss += loss
    avg_loss = cumul_loss/splits
    return w, avg_loss

# Find the best fit model.
def find_best_fit(x, y, reg):
    x = x[:,-2:]
    w_best = np.zeros((x.shape[1], 1))
    loss_smallest = np.inf
    # Try all models of degree 2 to 10.
    for i in range(2, 10):
        x = np.concatenate((np.array([x[:, -2]]).T ** i, x), axis=1)
        w, loss = cross_validation(x, y, reg)
        if loss < loss_smallest:
            loss_smallest = loss
            w_best = w
    return w_best

# Plot the data and the best fit model.
def plot_data(x, y, w, w_gt, stdev, x_point, y_point):
    fig = plt.figure(figsize=(6,6))
    plt.title('Data')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.grid()

    # Plot the data.
    plt.plot(x[:,-2], y, 'o', color='grey', markerfacecolor='none', label=f'Noisy Data, stdev={stdev}')

    # Plot the ground truth model.
    x_gt = np.array([[i] for i in np.linspace(start=-1.5, stop=1.5, num=100)])
    y_gt = np.array([np.polyval(w_gt, i) for i in x_gt])
    plt.plot(x_gt, y_gt, 'r--', label='Ground Truth Model')

    # Plot the best fit model.
    x_bf = np.array([[i] for i in np.linspace(start=-1.5, stop=1.5, num=100)])
    y_bf = np.array([np.polyval(w, i) for i in x_bf])
    plt.plot(x_bf, y_bf, 'b-', label='Best Fit Model')

    # Plot the test points.
    plt.plot(x_point[:,-2], y_point, 'o', color='black', markerfacecolor='purple', label=f'Test Data, stdev={stdev}')

    plt.legend()
    plt.show()

# Find the gradient of the loss function.
def grad(x, y, w, reg):
    return x.T@x@w - x.T@y + reg*w

# Find the loss function.
def loss(x, y, w, reg):
    return 1/2*npla.norm(x@w - y)**2 + 1/2*reg*npla.norm(w)**2

# Find the gradient descent solution for w.
def grad_descent(x, y, reg, lr=0.01, max_iter=1000, tol=1e-6):
    w = np.zeros((x.shape[1], 1))
    for i in range(max_iter):
        w_new = w - lr*grad(x, y, w, reg)
        if npla.norm(w_new - w) < tol:
            break
        w = w_new
    return w

# Find the stochastic gradient descent solution for w.
def sgd(x, y, reg=0, lr=0.01, max_iter=1000, tol=1e-6):
    w = np.zeros((x.shape[1], 1))
    for i in range(max_iter):
        for j in range(x.shape[0]):
            w_new = w - lr*grad(x[[j],:], y[[j],:], w, reg)
            if npla.norm(w_new - w) < tol:
                break
            w = w_new
    return w

