import numpy as np
from random import randint

import plot

def tanh(x):
    return np.tanh(x)

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def truth(a, b):
    sum = np.binary_repr(num=(a+b), width=8)
    sum = np.fromstring(sum, 'u1') - ord('0')
    sum = np.array(sum)
    sum = np.flip(sum)

    return sum

def y2two(sum):
    out = np.zeros([2, 8])
    for i in range(8):
        out[sum[i], i] = 1
    return out

def generate_input(a_o, b_o):

    a = np.binary_repr(num=a_o, width=8)
    a = np.fromstring(a, 'u1') - ord('0')
    a = np.array(a)
    a = np.flip(a)
    b = np.binary_repr(num=b_o, width=8)
    b = np.fromstring(b, 'u1') - ord('0')
    b = np.array(b)
    b = np.flip(b)

    two = np.append(a[np.newaxis, :],b[np.newaxis, :],axis=0)

    return two


class RNN():
    def __init__(self):
        super(RNN, self).__init__()
        rs = np.random.RandomState(54760)
        self.W = rs.uniform(-1, 1, (16, 16))
        self.U = rs.uniform(-1, 1, (16, 2))
        self.V = rs.uniform(-1, 1, (2, 16))
        self.b = rs.uniform(-1, 1, (16, 1))
        self.c = rs.uniform(-1, 1, (2, 1))
        self.h = rs.uniform(-1, 1, (16, 8))
        self.x = 0
        self.y = 0
        self.y_hat = 0

    def forward(self, x, ground_truth):
        """
        x: 2x8
        a: 16x1
        h: 16x8
        o: 2x8
        y: 2x8

        #weight
        W: 16x16
        U: 16x2
        V: 2x16
        b: 16x1
        c: 2x1

        a = b + W@h + U@x
        h = tanh(a)
        o = c + V@h
        y = softmax(o)
        """

        for i in range(8):
            if(i==0):
                a = self.b + self.U @ x[:,[i]]
            else:
                a = self.b + self.W @ self.h[:,[i-1]] + self.U @ x[:,[i]]
            self.h[:,i] = np.reshape(tanh(a),[16])
            o = self.c + self.V @ self.h[:,[i]]
            y = softmax(o)
            if(i==0):
                result = y
            else:
                result = np.concatenate((result, y), axis=1)

        self.x = x
        self.y = ground_truth
        self.y_hat = result

        return result

    def backward(self, alpha):
        o_g = self.y_hat - self.y
        h_g = np.zeros((16, 8))
        W_g = np.zeros((16, 16))
        U_g = np.zeros((16, 2))
        V_g = np.zeros((2, 16))
        b_g = np.zeros((16, 1))
        c_g = np.zeros((2, 1))

        self.h_origin = self.h.copy()

        for i in range(7, -1, -1):
            o_g_i = o_g[:,[i]]

            if(i==7):
                h_g[:,i] = np.reshape(self.V.T@o_g_i,[16])
            else:
                H2 = np.diag(1 - self.h_origin[:,i+1] ** 2)
                h_g[:,i] = np.reshape((self.W.T @ H2 @ h_g[:,[i+1]] + self.V.T @ o_g_i), [16])

            H = np.diag(1 - self.h_origin[:,i] ** 2)
            h_g_i = h_g[:,[i]]

            if(i!=0):
                W_g += (H @ h_g_i @ self.h_origin[:,[i-1]].T)
            U_g += (H @ h_g_i @ self.x[:,[i]].T)
            V_g += (o_g_i @ self.h_origin[:,[i]].T)
            b_g += (H @ h_g_i)
            c_g += o_g_i

        self.W -= (alpha*W_g)
        self.U -= (alpha*U_g)
        self.V -= (alpha*V_g)
        self.b -= (alpha*b_g)
        self.c -= (alpha*c_g)

def train(epoch, iterations, alpha):
    net = RNN()
    plot_acc = []
    for epoch_i in range(1,epoch+1):
        correct = 0
        error_digit = 0
        for iter_i in range(iterations):
            # calculate data
            a = randint(0, 127)
            b = randint(0, 127)
            x = generate_input(a, b)
            ground_truth = truth(a, b)
            y = y2two(ground_truth)
            y_hat = RNN.forward(net, x, y)

            # update
            RNN.backward(net, alpha)

            # correctness
            y_hat = np.argmax(y_hat, 0)
            right_digit = sum((ground_truth==y_hat).astype(int))
            if(right_digit==8):
                correct+=1
            error_digit += (8-right_digit)

        error = error_digit/(iterations*8)
        accuracy = correct/iterations
        plot_acc.append(accuracy*100)
        print('epoch ', epoch_i, ', error:', error, ', accuracy:', accuracy)

    plot.show_result('RNN', 0, epoch, 1, 0, 100, 20, plot_acc)

    return net

def test(net, num1, num2):
    x = generate_input(num1, num2)
    y = y2two(truth(num1, num2))
    y_hat = RNN.forward(net, x, y)

    y_hat = np.argmax(y_hat, 0)
    y_hat = np.flip(y_hat)
    y_hat = np.array2string(y_hat, separator='')
    y_hat = y_hat.replace("[", "")
    y_hat = y_hat.replace("]", "")
    y_hat = int(y_hat, 2)

    return y_hat