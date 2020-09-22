import numpy as np
import matplotlib.pyplot as plt
import random

np.random.seed(12)
num_observations = 500
x1 = np.random.multivariate_normal([0, 0], [[1, .75], [.75, 1]], num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, .75], [.75, 1]], num_observations)

X = np.vstack((x1, x2)).astype(np.float32)
Y = np.hstack((np.zeros(num_observations), np.ones(num_observations)))

fig, ax = plt.subplots()
for i in range(len(X)):
    if Y[i] == 1:
        ax.scatter(X[i][0], X[i][1], color='blue', marker='.')
    else:
        ax.scatter(X[i][0], X[i][1], color='red', marker='.')
plt.show()
X_1 = np.hstack((X, np.ones([2 * num_observations, 1])))
W = np.random.randn(1, 3)
print(W)
r = 0.01
s = 4e-3
k = 0
Y_1 = []
print(Y)
for j in range(10):
    error = 0
    for i in range(len(X_1)):
        y = np.dot(X_1[i], W.T)
        print(y)
        if y > 0:
            predict = 1
        else:
            predict = 0
        W = W + r * (Y[i] - predict) * X_1[i]
        k = k + 1
        print(W)
        error += abs(Y[i] - predict)
    print(error)
    if error / len(X_1) < s:
        break
print(k)

fig, ax = plt.subplots()
for i in range(len(X)):
    if Y[i] == 1:
        ax.scatter(X[i][0], X[i][1], color='blue', marker='.')
    else:
        ax.scatter(X[i][0], X[i][1], color='red', marker='.')
x = np.linspace(-5, 5, 100)
y = -W[0][0] / W[0][1] * x - W[0][2] / W[0][1]
plt.plot(x, y, linestyle='-')
plt.show()
