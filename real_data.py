import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def sigmoid_activation(x):
    sig = 1.0/(1+np.exp(-x))
    # print('sig ', sig)
    maximum = np.argmax(sig, axis=1)
    maximum = maximum.reshape((maximum.shape[0], 1))
    # print('maximum new : ', maximum, maximum.shape)
    return maximum


def predict(X, W):
    preds = sigmoid_activation(X.dot(W))
    # print('preds :  ', preds)
    # print(type(preds))
    # print(preds.shape)
    # preds[preds<=0.5] = 0
    # preds[preds>0.5] = 1
    return preds


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


fil = 'files/data_batch_1'
test_file = 'files/test_batch'
# dic = {}
dic = unpickle(fil)
test_dic = unpickle(test_file)

# (X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
X = dic[b'data']
y = np.array(dic[b'labels'])
y = y.reshape((y.shape[0], 1))


X = np.c_[X, np.ones(X.shape[0])]
print('shape x:  ', X.shape)
# (trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)

trainX = X
testX = test_dic[b'data']
testX = np.c_[testX, np.ones(testX.shape[0])]
trainY = y
testY = np.array(test_dic[b'labels'])
testY = testY.reshape((testY.shape[0], 1))

print("[INFO] training ...")
W = np.random.randn(trainX.shape[1], 10)
print(W.shape)
# print(W)
losses = []

epochs = int(input("enter epochs:"))
alpha = float(input("enter alpha:"))

for epoch in np.arange(0, epochs):
    preds = sigmoid_activation(X.dot(W))
    # print('preds:  ', preds)
    error = preds - trainY
    # print('shape x, y ', preds.shape, trainY.shape)
    # print('errors :  ', error)
    error[error != 0] = 1
    loss = np.sum(error)
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

# plt.style.use("ggplot")
# plt.figure()
# plt.title("Data")
# plt.scatter(testX[:], marker='o', s=30)

plt.style.use("ggplot")
plt.figure()
plt.title("Training Loss")
plt.plot(np.arange(0, epochs), losses)
plt.xlabel("Epochs #")
plt.ylabel("Loss")
plt.show()




