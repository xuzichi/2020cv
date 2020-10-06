import matplotlib.pyplot as plt
import sklearn.datasets
import numpy as np
import sklearn.linear_model
import matplotlib
import sys
import mnist
import random


def initialize(input_num, layer1_num, layer2_num, layer3_num, layer4_num, out_num):
    model = dict()
    np.random.seed(12)
    w1 = np.random.randn(input_num, layer1_num)/np.sqrt(input_num)
    b1 = np.random.randn(1, layer1_num)
    w2 = np.random.randn(layer1_num, layer2_num)/np.sqrt(layer1_num)
    b2 = np.random.randn(1, layer2_num)
    w3 = np.random.randn(layer2_num, layer3_num)/np.sqrt(layer2_num)
    b3 = np.random.randn(1,layer3_num)
    w4 = np.random.randn(layer3_num, layer4_num) / np.sqrt(layer3_num)
    b4 = np.random.randn(1, layer4_num)
    w5 = np.random.randn(layer4_num, out_num) / np.sqrt(layer4_num)
    b5 = np.random.randn(1, out_num)
    model['w1'] = w1
    model['b1'] = b1
    model['w2'] = w2
    model['b2'] = b2
    model['w3'] = w3
    model['b3'] = b3
    model['w4'] = w4
    model['b4'] = b4
    model['w5'] = w5
    model['b5'] = b5
    return model


def tanhPrime(x):
    x = 1 - np.power(x, 2)
    return x


def sigmoidPrime(x):
    x = x * (1 - x)
    return x


def ReLuPrime(x):
    # ReLu 导数
    x = np.where(x > 0, 1, 0)
    return x


def ReLu(x):
    # ReLu 函数
    x = (np.abs(x) + x) / 2.0
    return x


def predict(model, x):
    w1, b1, w2, b2, w3, b3, w4, b4, w5, b5 = model['w1'], model['b1'], model['w2'], model['b2'], model['w3'], model['b3'], model['w4'], model['b4'], model['w5'], model['b5']

    z1 = x.dot(w1) + b1
    # a1 = np.tanh(z1)
    # a1 = 1 / (1 + np.exp(z1))
    a1 = ReLu(z1)
    z2 = a1.dot(w2) + b2
    # a2 = np.tanh(z2)
    # a2 = 1 / (1 + np.exp(z2))
    a2 = ReLu(z2)
    z3 = a2.dot(w3) + b3
    # a3 = np.tanh(z3)
    # a3 = 1 / (1 + np.exp(z3))
    a3 = ReLu(z3)
    z4 = a3.dot(w4) + b4
    # a4 = np.tanh(z4)
    # a4 = 1 / (1 + np.exp(z4))
    a4 = ReLu(z4)
    z5 = a4.dot(w5) + b5
    exp_scores = np.exp(z5)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)


def forward(model, x, y):
    epsilon = 1e-5
    reg_lamda = 1e-4
    w1, b1, w2, b2, w3, b3, w4, b4, w5, b5 = model['w1'], model['b1'], model['w2'], model['b2'], model['w3'],model['b3'], model['w4'], model['b4'], model['w5'], model['b5']

    z1 = x.dot(w1) + b1
    # a1 = np.tanh(z1)
    # a1 = 1 / (1 + np.exp(z1))
    a1 = ReLu(z1)
    z2 = a1.dot(w2) + b2
    # a2 = np.tanh(z2)
    # a2 = 1 / (1 + np.exp(z2))
    a2 = ReLu(z2)
    z3 = a2.dot(w3) + b3
    # a3 = np.tanh(z3)
    # a3 = 1 / (1 + np.exp(z3))
    a3 = ReLu(z3)
    z4 = a3.dot(w4) + b4
    # a4 = np.tanh(z4)
    # a4 = 1 / (1 + np.exp(z4))
    a4 = ReLu(z4)
    z5 = a4.dot(w5) + b5
    print(z5)
    exp_scores = np.exp(z5)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # 分类输出
    #print(probs[0])
    loss = 0
    for i in range(probs.shape[0]):
        loss += -np.log(probs[i, y[i]])
    loss = loss/probs.shape[0]  # 最终的loss
    #导数信息
    delta6 = probs
    delta6[range(x.shape[0]), y] -= 1
    dw5 = (a4.T).dot(delta6)
    db5 = np.sum(delta6, axis=0, keepdims=True)

    delta5 = delta6.dot(w5.T) * ReLuPrime(a4)
    # delta5 = delta6.dot(w5.T) * tanhPrime(a4)
    # delta5 = delta6.dot(w5.T) * sigmoidPrime(a4)
    dw4 = np.dot(a3.T, delta5)
    db4 = np.sum(delta5, axis=0)

    delta4 = delta5.dot(w4.T) * ReLuPrime(a3)
    # delta4 = delta5.dot(w4.T) * tanhPrime(a3)
    # delta4 = delta5.dot(w4.T) * sigmoidPrime(a3)
    dw3 = np.dot(a2.T, delta4)
    db3 = np.sum(delta4, axis=0)

    delta3 = delta4.dot(w3.T) * ReLuPrime(a2)
    # delta3 = delta4.dot(w3.T) * tanhPrime(a2)
    # delta3 = delta4.dot(w3.T) * sigmoidPrime(a2)
    dw2 = np.dot(a1.T, delta3)
    db2 = np.sum(delta3, axis=0)

    delta2 = delta3.dot(w2.T) * ReLuPrime(a1)
    # delta2 = delta3.dot(w2.T) * tanhPrime(a1)
    # delta2 = delta3.dot(w2.T) * sigmoidPrime(a1)
    dw1 = np.dot(x.T, delta2)
    db1 = np.sum(delta2, axis=0)


    dw5 += reg_lamda * w5
    dw4 += reg_lamda * w4
    dw3 += reg_lamda * w3
    dw2 += reg_lamda * w2
    dw1 += reg_lamda * w1


    w1 += -epsilon * dw1
    b1 += -epsilon * db1
    w2 += -epsilon * dw2
    b2 += -epsilon * db2
    w3 += -epsilon * dw3
    b3 += -epsilon * db3
    w4 += -epsilon * dw4
    b4 += -epsilon * db4
    w5 += -epsilon * dw5
    b5 += -epsilon * db5

    model['w1'] = w1
    model['b1'] = b1
    model['w2'] = w2
    model['b2'] = b2
    model['w3'] = w3
    model['b3'] = b3
    model['w4'] = w4
    model['b4'] = b4
    model['w5'] = w5
    model['b5'] = b5
    return loss, model



def main():
    epochs = 201
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()

    test_images = mnist.test_images()
    test_labels = mnist.test_labels()

    n_train, w, h = train_images.shape
    X_train = train_images.reshape((n_train, w * h))  # 维度为60000 * 784
    Y_train = train_labels  # 60000 * 1个label

    n_test, w, h = test_images.shape
    X_test = test_images.reshape((n_test, w * h))
    Y_test = test_labels

    model = initialize(784, 128, 64, 32, 16, 10)  # 初始化模型参数
    batch_size = 600
    train_error = []
    test_acc = []
    for epoch in range(epochs):
        loss_ep = 0

        index = np.arange(60000)
        np.random.shuffle(index)
        X_train = X_train[index, :]
        Y_train = Y_train[index]

        for i in range(100):
            begin_num = i * batch_size
            end_num = i * batch_size + batch_size
            loss, model = forward(model, X_train[begin_num:end_num, :], Y_train[begin_num:end_num])
            loss_ep += loss
        loss_ep = loss_ep / batch_size
        train_error.append(loss_ep)

        y_predict = predict(model, X_test)
        counter = 0
        for i in range(len(y_predict)):
            if y_predict[i] == Y_test[i]:
                counter += 1
        test_acc.append(counter/len(y_predict))
        if epoch % 10 == 0:
            print('epoch:', epoch, 'loss:', loss_ep, 'test_acc:', counter/len(y_predict), '\n')

    plt.title(" training error curves")
    plt.xlabel("epochs")
    plt.ylabel("loss_average")
    x = np.arange(0, epochs)
    plt.plot(x, train_error)
    plt.show()

    plt.title(" test acc curves")
    plt.xlabel("epochs")
    plt.ylabel("loss_average")
    x = np.arange(0, epochs)
    plt.plot(x, test_acc)
    plt.show()


if __name__ == '__main__':
    main()
