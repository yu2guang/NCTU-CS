import numpy as np
from numpy.random import RandomState

# generate data function
def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)


def generate_XOR_easy():
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1 * i, 0.1 * i])
        labels.append(0)

        if 0.1 * i == 0.5:
            continue

        inputs.append([0.1 * i, 1 - 0.1 * i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21, 1)


# sigmoid calculation
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def derivative_sigmoid(x):
    return np.multiply(x, 1.0 - x)


# plot
def show_result(x, y, pred_y):
    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.show()


# training code
def train(inputs, labels, n, LR):
    rs = np.random.RandomState(547601234)
    w1 = rs.uniform(-1, 1, (2, 2))
    w2 = rs.uniform(-1, 1, (2, 2))
    w3 = rs.uniform(-1, 1, (2, 1))
    b1 = rs.uniform(-1, 1, (1, 2))
    b2 = rs.uniform(-1, 1, (1, 2))
    b3 = rs.uniform(-1, 1, (1, 1))

    epoch_print = False
    epoch = 1
    while(1):
        if (epoch % 100 == 0):
            epoch_print = True
            y_origin = []
            y_hat = []
        b3_t = 0
        w3_t = 0
        b2_t = 0
        w2_t = 0
        b1_t = 0
        w1_t = 0
        for i in range(n):

            x_i = inputs[i]  # 1x2
            y_i = labels[i]  # 1

            """forward pass"""
            # Layer 1
            z1_bs = x_i @ w1 + b1
            z1 = sigmoid(z1_bs)  # 1x2
            # Layer 2
            z2_bs = z1 @ w2 + b2
            z2 = sigmoid(z2_bs)  # 1x2
            # Layer 3
            y_bs = z2 @ w3 + b3
            y = sigmoid(y_bs)  # 1

            if (epoch_print):
                y_origin.append(y.reshape(-1))

            """back propagation"""
            ## compute gradient
            # Layer 3
            b3_g = (y-y_i) * derivative_sigmoid(y)  # 1x1
            w3_g = (b3_g * z2).T  # 2x1
            # Layer 2
            b2_g = b3_g * w3.T * derivative_sigmoid(z2)  # 1x2
            w2_g = np.tile(b2_g, (2, 1)) * np.tile(z1, (2, 1)).T  # 2x2
            # Layer 1
            b1_g = b2_g @ w2 * derivative_sigmoid(z1)  # 1x2
            w1_g = np.tile(b1_g, (2, 1)) * np.tile(x_i, (2, 1)).T  # 2x2

            b3_t += b3_g
            w3_t += w3_g
            b2_t += b2_g
            w2_t += w2_g
            b1_t += b1_g
            w1_t += w1_g

        # update weights
        b3 = b3 - LR * b3_t
        w3 = w3 - LR * w3_t
        b2 = b2 - LR * b2_t
        w2 = w2 - LR * w2_t
        b1 = b1 - LR * b1_t
        w1 = w1 - LR * w1_t

        if (epoch_print):
            y_origin = np.array(y_origin)
            """MSE"""
            mse = np.mean((labels.reshape(-1)-y_origin.reshape(-1))**2)

            """Accuracy"""
            y_hat = (y_origin > 0.5).astype(np.int)
            right = (labels[:, 0] == y_hat[:, 0]).astype(np.int)
            accuracy = np.count_nonzero(right) / n
            print('epoch ', epoch,', loss:', mse,', accuracy:', accuracy)

            """Final result"""
            if(accuracy==1):
                print(y_origin)
                break

            epoch_print = False

        epoch += 1

    return y_hat

## start training
# linear
x, y = generate_linear(n=100)
pred_y = train(x, y, 100, 0.1)
show_result(x, y, pred_y)
# XOR
x, y = generate_XOR_easy()
pred_y = train(x, y, 21, 1)
show_result(x, y, pred_y)