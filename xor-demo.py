
import numpy as np
import matplotlib.pyplot as plt

N = 4
D = 2

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# targets
T = np.array([0, 1, 1, 0])

ones = np.array([[1]*N]).T

# as visualized, this is a 2d problem. 
plt.scatter(X[:, 0], X[:, 1], c=T)
plt.show()

# to solve XOR problems in logistic regression, convert to 3D


# add a column of xy = x*y
xy = (X[:,0] * X[:,1]).reshape(N, 1)
Xb = np.concatenate((ones, xy, X), axis=1)

w = np.random.randn(D + 2)

z = Xb.dot(w)

def sigmoid(z):
	return 1 / ( 1 + np.exp(-z) )

Y = sigmoid(z)

# calculate the cross-entropy
def cross_entropy(T, Y):
	E = 0
	for i in range(N):
		if T[i] == 1:
			E -= np.log(Y[i])
		else:
			E -= np.log(1 - Y[i])

	return E

print( cross_entropy(T, Y) )

learning_rate = 0.001
error = []

for i in range(5000):
    e = cross_entropy(T, Y)
    error.append(e)
    if i % 100 == 0:
        print(e)
    # gradient descent weight update with regularization
    w += learning_rate * ( np.dot((T - Y).T, Xb ) - 0.01*w  )

    # recalculate Y
    Y = sigmoid(Xb.dot(w))

plt.plot(error)
plt.title("Cross-entropy per iteration")
plt.show()

print("Final w:", w)
print("Final classification rate:", 1 - np.abs(T - np.round(Y)).sum() / N )





