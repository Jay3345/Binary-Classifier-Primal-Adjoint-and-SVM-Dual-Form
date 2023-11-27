import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from cvxopt import matrix, solvers
import pandas as pd


########################
### HELPER FUNCTIONS ###((We did not write these))
########################

def _plotCls():
    n = 100
    lamb = 0.01
    gen_model = 1
    kernel_func = lambda X1, X2: polyKernel(X1, X2, 2)

    # Generate data
    Xtrain, ytrain = generateData(n=n, gen_model=gen_model)
    # Learn and plot results

    # Primal
    print("minBinDev: ")
    w, w0 = minBinDev(Xtrain, ytrain, lamb)
    plotModel(Xtrain, ytrain, w, w0, classify)
    # Adjoint
    print("Adjoint Hinge")
    a, a0 = adjHinge(Xtrain, ytrain, lamb, kernel_func)
    plotAdjModel(Xtrain, ytrain, a, a0, kernel_func, adjClassify)
    # Dual
    print("Dual Hinge")
    a, b = dualHinge(Xtrain, ytrain, lamb, kernel_func)
    plotDualModel(Xtrain, ytrain, a, b, lamb, kernel_func, dualClassify)

def linearKernel(X1, X2):
    return X1 @ X2.T


def polyKernel(X1, X2, degree):
    return (X1 @ X2.T + 1) ** degree


def gaussKernel(X1, X2, width):
    distances = cdist(X1, X2, 'sqeuclidean')
    return np.exp(- distances / (2*(width**2)))


def generateData(n, gen_model):

    # Controlling the random seed will give you the same 
    # random numbers every time you generate the data. 
    # The seed controls the internal random number generator (RNG).
    # Different seeds produce different random numbers. 
    # This can be handy if you want reproducible results for debugging.
    # For example, if your code *sometimes* gives you an error, try
    # to find a seed number (0 or others) that produces the error. Then you can
    # debug your code step-by-step because every time you get the same data.

    # np.random.seed(0)  # control randomness when debugging

    if gen_model == 1 or gen_model == 2:
        # Gen 1 & 2
        d = 2
        w_true = np.ones([d, 1])

        X = np.random.randn(n, d)

        if gen_model == 1:
            y = np.sign(X @ w_true)  # generative model 1
        else:
            y = np.sign((X ** 2) @ w_true - 1)  # generative model 2

    elif gen_model == 3:
        # Gen 3
        X, y = generateMoons(n)

    else:
        raise ValueError("Unknown generative model")

    return X, y


def generateMoons(n, noise=0.1):
    n_samples_out = n // 2
    n_samples_in = n - n_samples_out
    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5

    X = np.vstack(
        [np.append(outer_circ_x, inner_circ_x), 
         np.append(outer_circ_y, inner_circ_y)]
    ).T
    X += np.random.randn(*X.shape) * noise

    y = np.hstack(
        [-np.ones(n_samples_out, dtype=np.intp), 
         np.ones(n_samples_in, dtype=np.intp)]
    )[:, None]
    return X, y


def plotPoints(X, y):
    # plot the data points from two classes
    X0 = X[y.flatten() >= 0]
    X1 = X[y.flatten() < 0]

    plt.scatter(X0[:, 0], X0[:, 1], marker='x', label='class -1')
    plt.scatter(X1[:, 0], X1[:, 1], marker='o', label='class +1')
    return


def getRange(X):
    x_min = np.amin(X[:, 0]) - 0.1
    x_max = np.amax(X[:, 0]) + 0.1
    y_min = np.amin(X[:, 1]) - 0.1
    y_max = np.amax(X[:, 1]) + 0.1
    return x_min, x_max, y_min, y_max


def plotModel(X, y, w, w0, classify):

    plotPoints(X, y)

    # plot model
    x_min, x_max, y_min, y_max = getRange(X)
    grid_step = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),
                         np.arange(y_min, y_max, grid_step))
    z = classify(np.c_[xx.ravel(), yy.ravel()], w, w0)

    # Put the result into a color plot
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=plt.cm.RdBu, alpha=0.5)
    plt.legend()
    plt.show()
    return


def plotAdjModel(X, y, a, a0, kernel_func, adjClassify):

    plotPoints(X, y)

    # plot model
    x_min, x_max, y_min, y_max = getRange(X)
    grid_step = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),
                         np.arange(y_min, y_max, grid_step))
    z = adjClassify(np.c_[xx.ravel(), yy.ravel()], a, a0, X, kernel_func)

    # Put the result into a color plot
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=plt.cm.RdBu, alpha=0.5)
    plt.legend()
    plt.show()
    return


def plotDualModel(X, y, a, b, lamb, kernel_func, dualClassify):

    plotPoints(X, y)

    # plot model
    x_min, x_max, y_min, y_max = getRange(X)
    grid_step = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),
                         np.arange(y_min, y_max, grid_step))
    z = dualClassify(np.c_[xx.ravel(), yy.ravel()], a, b, X, y, 
                     lamb, kernel_func)

    # Put the result into a color plot
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=plt.cm.RdBu, alpha=0.5)
    plt.legend()
    plt.show()

    return


def plotDigit(x):
    img = x.reshape((28, 28))
    plt.imshow(img, cmap='gray')
    plt.show()
    return


########################
########################
########################

#########################
## My(And Alex's) Work ##
#########################


# Question 1a
def minBinDev(X, y, lamb):
    d = X.shape[1]

    # Define objective
    def obj_func(u):
        w0 = u[-1]  # last unknown is w0
        w = u[:-1]  # the first d dimensions are w
        w = w[:, None]  # make it d-by-1

        loss = np.sum(np.logaddexp(0, -y * (X @ w + w0)))
        reg = 0.5 * lamb * np.sum(np.square(w))

        return loss + reg

    # Initial guess of unknowns, shouldn't matter for convex problem
    u0 = np.ones(d + 1)  # first d dimension for w, and last for w0

    sol = minimize(obj_func, u0)  # objective function + initial guess as inputs

    # Get the solution
    w = sol['x'][:-1][:, None]  # make it d-by-1
    w0 = sol['x'][-1]

    return w, w0



# Question 1b
def minHinge(X, y, lamb, stabilizer=1e-5):
    n = X.shape[0]
    d = X.shape[1]
    diagY = np.eye(n) * y

    # Make q
    q = np.zeros(d+1)
    q = np.append(q, np.ones(n))

    # Make G
    G11 = np.zeros([n, d])
    G12 = np.zeros([n, 1])
    G13 = np.negative(np.identity(n))
    G1 = np.concatenate([G11, G12, G13], axis=1) # Concatenate on x-axis

    G21 = -1 * diagY @ X
    G22 = -1 * y
    G23 = np.negative(np.identity(n))
    G2 = np.concatenate([G21, G22, G23], axis=1)

    G = np.concatenate([G1, G2], axis=0) # Concatenate on y-axis

    # Make h
    h1 = np.zeros(n)
    h2 = np.negative(np.ones(n))
    h = np.concatenate([h1, h2], axis=0)

    # Make P
    P11 = lamb * np.identity(d)
    P12 = np.zeros([d, 1])
    P13 = np.zeros([d, n])
    P21 = np.zeros([1, d])
    P22 = np.zeros([1, 1])
    P23 = np.zeros([1, n])
    P31 = np.zeros([n, d])
    P32 = np.zeros([n, 1])
    P33 = np.zeros([n, n])
    P1 = np.concatenate([P11, P12, P13], axis=1)
    P2 = np.concatenate([P21, P22, P23], axis=1)
    P3 = np.concatenate([P31, P32, P33], axis=1)
    P = np.concatenate([P1, P2, P3], axis=0)
    P = P + stabilizer * np.eye(n+d+1)

    # Find solution
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h)

    w = sol['x'][:2] # make it d-by-1
    w0 = sol['x'][-1]

    return w, w0


# Question 1c
def classify(Xtest, w, w0):
    yhat = np.sign(Xtest @ w + w0)
    return yhat


# Question 1d
def synExperimentsRegularize():
    n_runs = 100
    n_train = 100
    n_test = 1000
    lamb_list = [0.001, 0.01, 0.1, 1.]
    gen_model_list = [1, 2, 3]

    train_acc_bindev = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    test_acc_bindev = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    train_acc_hinge = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    test_acc_hinge = np.zeros([len(lamb_list), len(gen_model_list), n_runs])

    for r in range(n_runs):
        for i, lamb in enumerate(lamb_list):
            for j, gen_model in enumerate(gen_model_list):
                Xtrain, ytrain = generateData(n=n_train, gen_model=gen_model)
                Xtest, ytest = generateData(n=n_test, gen_model=gen_model)

                w, w0 = minBinDev(Xtrain, ytrain, lamb)
                train_pred_bindev = classify(Xtrain, w, w0)
                test_pred_bindev = classify(Xtest, w, w0)
                train_acc_bindev[i, j, r] = np.mean(train_pred_bindev == ytrain)
                test_acc_bindev[i, j, r] = np.mean(test_pred_bindev == ytest)

                w, w0 = minHinge(Xtrain, ytrain, lamb)
                w = np.array(w) # w is returned as cvxopt matrix. Turn it back to numpy array
                train_pred_hinge = classify(Xtrain, w, w0)
                test_pred_hinge = classify(Xtest, w, w0)
                train_acc_hinge[i, j, r] = np.mean(train_pred_hinge == ytrain)
                test_acc_hinge[i, j, r] = np.mean(test_pred_hinge == ytest)

    # compute the average accuracies over runs
    train_acc_bindev = np.mean(train_acc_bindev, axis=2)
    test_acc_bindev = np.mean(test_acc_bindev, axis=2)
    train_acc_hinge = np.mean(train_acc_hinge, axis=2)
    test_acc_hinge = np.mean(test_acc_hinge, axis=2)

    # combine accuracies
    acc_train=np.hstack((train_acc_bindev, train_acc_hinge))
    acc_test=np.hstack((test_acc_bindev, test_acc_hinge))

    # return 4-by-6 train accuracy and 4-by-6 test accuracy
    return acc_train, acc_test



# Question 2a
def adjBinDev(X, y, lamb, kernel_function):
    n,d = X.shape
    K = kernel_function(X,X)
    # Define objective
    # Option 1: define as a function
    def obj_func(u):
        a0 = u[-1]  # last unknown is a0
        a = u[:-1]  # the first n dimensions are a
        a = a[:, None]  # make it n-by-1

        loss = np.sum(np.logaddexp(0, (- y * (K.T @ a + a0))))
        reg = 0.5 * lamb * a.T @ K @ a

        return loss + reg

    # Initial guess of unknowns, doesn't matter for convex problem
    u0 = np.ones(n + 1)  # first n dimension for a, and last for a0

    sol = minimize(obj_func, u0)  # objective function + initial guess as inputs

    # Get the solution
    a = sol['x'][:-1][:, None]  # make it n-by-1
    a0 = sol['x'][-1]
    return a, a0


# Question 2b
def adjHinge(X, y, lamb, kernel_func, stabilizer=1e-5):
    n,d =X.shape
    diagY = np.diagflat(y)
    K=kernel_func(X, X)

    # Make q
    q = np.zeros(n+1)
    q = np.append(q, np.ones(n))

    # Make G
    G11 = np.zeros((n, n))
    G12 = np.zeros((n, 1))
    G13 = -(np.identity(n))
    G21 = -(diagY * K)
    G22 = -y
    G23 = -(np.identity(n))

    G1 = np.concatenate([G11, G12, G13], axis=1) # Concatenate on x-axis
    G2 = np.concatenate([G21, G22, G23], axis=1)
    G = np.concatenate([G1, G2], axis=0) # Concatenate on y-axis

    # Make h
    h1 = np.zeros(n)
    h2 = -(np.ones(n))
    h = np.hstack((h1, h2))

    # Make P
    P11 = lamb * np.identity(n)
    P12 = np.zeros([n, 1])
    P13 = np.zeros([n, n])
    P21 = np.zeros([1, n])
    P22 = np.zeros([1, 1])
    P23 = np.zeros([1, n])
    P31 = np.zeros([n, n])
    P32 = np.zeros([n, 1])
    P33 = np.zeros([n, n])
    P1 = np.concatenate([P11, P12, P13], axis=1)
    P2 = np.concatenate([P21, P22, P23], axis=1)
    P3 = np.concatenate([P31, P32, P33], axis=1)
    P = np.concatenate([P1, P2, P3], axis=0)
    P = P + stabilizer * np.eye(n+n+1)

    # Find solution
    solvers.options['show_progress'] = False
    sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
    a = np.array(sol['x'][:n])
    a0 = np.array(sol['x'][n])

    return a, a0


# Question 2c
def adjClassify(Xtest, a, a0, X, kernel_func):
    K=kernel_func(Xtest, X)
    return np.sign(K @ a + a0)


# Question 2d
def synExperimentsKernel():
    n_runs = 10
    n_train = 100
    n_test = 1000
    lamb = 0.001
    kernel_list = [linearKernel,
                lambda X1, X2: polyKernel(X1, X2, 2),
                lambda X1, X2: polyKernel(X1, X2, 3),
                lambda X1, X2: gaussKernel(X1, X2, 1.0),
                lambda X1, X2: gaussKernel(X1, X2, 0.5)]
    gen_model_list = [1, 2, 3]
    train_acc_bindev = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    test_acc_bindev = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    train_acc_hinge = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    test_acc_hinge = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    for r in range(n_runs):
        for i, kernel in enumerate(kernel_list):
            for j, gen_model in enumerate(gen_model_list):
                Xtrain, ytrain = generateData(n=n_train, gen_model=gen_model)
                Xtest, ytest = generateData(n=n_test, gen_model=gen_model)

                a, a0 = adjBinDev(Xtrain, ytrain, lamb, kernel_list[i])
                train_pred_bindev = adjClassify(Xtrain, a, a0,Xtrain,kernel_list[i])
                test_pred_bindev = adjClassify(Xtest, a, a0,Xtrain,kernel_list[i])

                train_acc_bindev[i, j, r] = np.mean(ytrain == train_pred_bindev)
                test_acc_bindev[i, j, r] = np.mean(ytest == test_pred_bindev)

                a, a0 = adjHinge(Xtrain, ytrain, lamb, kernel_list[i])

                train_pred_hinge = adjClassify(Xtrain, a, a0, Xtrain, kernel_list[i])
                test_pred_hinge = adjClassify(Xtest, a, a0, Xtrain, kernel_list[i])

                train_acc_hinge[i, j, r] = np.mean(ytrain == train_pred_hinge)
                test_acc_hinge[i, j, r] = np.mean(ytest == test_pred_hinge)

    train_acc_bindev = np.mean(train_acc_bindev, axis=2)
    test_acc_bindev = np.mean(test_acc_bindev, axis=2)
    train_acc_hinge = np.mean(train_acc_hinge, axis=2)
    test_acc_hinge = np.mean(test_acc_hinge, axis=2)

    acc_train=np.hstack((train_acc_bindev, train_acc_hinge))
    acc_test=np.hstack((test_acc_bindev, test_acc_hinge))

    return acc_train, acc_test



# Question 3a
def dualHinge(X, y, lamb, kernel_func, stabilizer=1e-5):
    n = X.shape[0]
    d = X.shape[1]
    K = kernel_func(X, X)
    deltaY = np.eye(n) * y

    # Construct G
    G1 = np.identity(n)
    G2 = -1 * np.identity(n)
    G = np.concatenate([G1, G2], axis=0)

    # Construct h
    h1 = np.ones(n)
    h2 = np.zeros(n)
    h = np.concatenate([h1, h2], axis=0)

    # Construct A
    A = y.T
    A = A.astype('float') # For some reason gen_model=3 graph doesn't work without this line

    # Construct b
    b = np.zeros(1)

    # Construct q
    q = -1 * np.ones(n)

    # Construct P
    P = (1/lamb) * deltaY @ K @ deltaY
    P = P + stabilizer * np.eye(n)

    # Find optimal alpha
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)
    a = np.array(sol['x'][:n])

    # Find b
    index = (np.abs(a - 0.5)).argmin()
    b = y[index] - (1/lamb) * (K.T)[index] @ deltaY @ a

    return a, b


# Question 3b
def dualClassify(Xtest, a, b, X, y, lamb, kernel_func):
    n = X.shape[0]
    deltaY = np.eye(n) * y
    return np.sign((1/lamb) * kernel_func(Xtest, X) @ deltaY @ a + b)


# Question 3c
def cvMnist(dataset_folder, lamb_list, kernel_list, k=5):
    train_data = pd.read_csv(f"{dataset_folder}/A2train.csv", header=None).to_numpy()
    X = train_data[:, 1:] / 255.
    y = train_data[:, 0][:, None]
    y[y == 4] = -1
    y[y == 9] = 1
    cv_acc = np.zeros([k, len(lamb_list), len(kernel_list)])

    # perform any necessary setup
    # Parition the X and y sets into k equal subsets
    Xpart = np.array_split(X, k)
    ypart = np.array_split(y, k)

    for i, lamb in enumerate(lamb_list):
        for j, kernel_func in enumerate(kernel_list):
            for l in range(k):
                # Extract the lth partition from the data
                removedX = Xpart.pop(l)
                removedy = ypart.pop(l)
                Xtrain = np.concatenate(Xpart)
                ytrain = np.concatenate(ypart)
                Xpart.insert(l, removedX)
                ypart.insert(l, removedy)

                # Get validation dataset
                Xval = Xpart[l]
                yval = ypart[l]

                # Train + Evaluate Accuracy
                a, b = dualHinge(Xtrain, ytrain, lamb, kernel_func)
                yhat = dualClassify(Xval, a, b, Xtrain, ytrain, lamb, kernel_func)
                cv_acc[l, i, j] = np.mean(yhat == yval)

    # compute the average accuracies over k folds
    avg_acc = np.mean(cv_acc, axis=0)
    # identify the best lamb and kernel function
    lambIndex = 0
    kernelIndex = 0
    largestAccuracy = float('-inf')

    for l in range(len(lamb_list)):
        for k in range(len(kernel_list)):
            if avg_acc[l][k] > largestAccuracy:
                largestAccuracy = avg_acc[l][k]
                lambIndex = l
                kernelIndex = k

    #       return a "len(lamb_list)-by-len(kernel_list)" accuracy variable,
    #       the best lamb and the best kernel
    if(k==0):
        return avg_acc, lamb_list[lambIndex], "Linear kernel"
    elif(k==1):
        return avg_acc, lamb_list[lambIndex], "Poly kernel"
    else:
        return avg_acc, lamb_list[lambIndex], "Gauss kernel"



def main():
    # Question 1d running + testing
    print("Question 1d running + testing (takes about 1 minute)")
    train_acc, test_acc = synExperimentsRegularize()
    print("training accuracy(explained in report)")
    print(train_acc)
    print("test accuracy(explained in report)")
    print(test_acc)
    print("\n\n")

    # Question 3a testing values
    print("Question 3a testing values")
    n = 100
    lamb = 0.01
    gen_model = 1
    kernel_func = lambda X1, X2: linearKernel(X1, X2)
    X, y = generateData(n, gen_model)
    deltaY = y * np.eye(n)

    w1, w0 = minHinge(X, y, lamb)
    a1, a0 = adjHinge(X, y, lamb, kernel_func)
    w2 = X.T @ a1 + a0
    a, b = dualHinge(X, y, lamb, kernel_func)

    w3 = (1/lamb) * (X.T @ deltaY @ a)

    print("Alpha 0 = "+ str(a0))
    print("Min hinge")
    print(w1)
    print("Adj hinge")
    print(w2)
    print()
    print("Dual hinge")
    print(w3)
    print("\n\n")

    # Question 2d testing
    print("testing 2d, should take about 6-8 minutes")
    train_acc, test_acc = synExperimentsKernel()
    print("training accuracies(explained in report)")
    print(train_acc)
    print("test accuracies(explained in report)")
    print(test_acc)
    print("\n\n")

    # Question 3c testing
    print("3c testing, should take about 40-50 seconds")
    avg_acc, best_lamb, best_kernel = cvMnist("CHANGE_THIS_TO_FILE_LOCATION", [0.001, 1.0], [linearKernel, lambda X1, X2: polyKernel(X1, X2, 3), lambda X1, X2: gaussKernel(X1, X2, 1.0)])

    print("Average accuracies(explained in report)")
    print(avg_acc)
    print("Best Lambda: "+str(best_lamb))
    print("Best kernel: " + best_kernel)
    print("\n\n")
  
    print("Printing Models: ")
    _plotCls()

if __name__ == "__main__":
    main()
