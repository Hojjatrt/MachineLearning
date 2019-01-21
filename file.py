import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def sigmoid_activation(x):
    return 1.0/(1+np.exp(-x))


def predict(X, W):
    preds = sigmoid_activation(X.dot(W))

    preds[preds<=0.5] = 0
    preds[preds>0.5] = 1
    return preds


(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))

X = np.c_[X, np.ones(X.shape[0])]

(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)

print("[INFO] training ...")
W = np.random.randn(X.shape[1], 1)
losses = []

epochs = int(input("enter epochs:"))
alpha = float(input("enter alpha:"))

for epoch in np.arange(0, epochs):
    preds = sigmoid_activation(trainX.dot(W))
    error = preds - trainY
    loss = np.sum(error**2)
    losses.append(loss)

    gradient = trainX.T.dot(error)
    W += -alpha * gradient
    if (epoch==0 or (epoch+1)%5==0):
        print("[INFO] epoch={}, loss={}".format(epoch, loss))

print("[INFO] evaluating...")
preds = predict(testX, W)
print(classification_report(testY, preds))


# while True:
#     Wgradient = evaluate_gradient(loss, data, W)
#     W += -alpha * Wgradient

import matplotlib.pyplot as plt

plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:,0], testX[:,1], marker='o', s=30)

plt.style.use("ggplot")
plt.figure()
plt.title("Training Loss")
plt.plot(np.arange(0, epochs), losses)
plt.xlabel("Epochs #")
plt.ylabel("Loss")
plt.show()




