import  numpy as np
import  pandas as pd
from matplotlib import pyplot as plt


# Open and Look at the data
data = pd.read_csv("mnist-digit-recognizer/train.csv")
data.head()

# changing the data to array for further working
data = np.array(data)
m, n = data.shape

np.random.shuffle(data)

# Splitting the data

data_dev = data[0:1000].T       #data used for checking predictions using our trained model
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T         #data used for training our model
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

# making functions for the model

def init_param():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return W1, b1, W2, b2


def ReLU(Z):
    return np.maximum(0, Z)


def SoftMax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = (W2).dot(A1) + b2
    A2 = SoftMax(Z2)
    return Z1, A1, Z2, A2


def one_hot(Y):
    # this function is used to generate a matrix that will be used to compare with our output which will be probabilities
    # this functions adds 1 to the index which is the actual digit and we have to minimize that difference (p - 1)^2

    one_hot_Y = np.zeros((Y.shape[0], Y.max() + 1))
    one_hot_Y[np.arange(Y.shape[0]), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def ReLU_deriv(Z):
    return Z > 0


def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    one_hot_Y = one_hot(Y)

    dZ2 = A2 - one_hot_Y
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = (1 / m) * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ1)

    return dW1, db1, dW2, db2


# Applying gradient descent for updating the parameters

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):  # alpha is the rate of gradient descent
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_param()

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

    predictions = get_predictions(A2)
    print(f"Accuracy: {get_accuracy(predictions, Y)}")
    print("Done!!")
    return W1, b1, W2, b2

W1, b1 , W2, b2 = gradient_descent(X_train, Y_train, 500, 0.2)

# Now making actual predictions too see if our model actually works
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions


def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

for i in range(4):
    test_prediction(i, W1, b1, W2, b2)