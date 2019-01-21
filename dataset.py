def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


import numpy as np
fil = 'files/data_batch_1'
test_file = 'files/test_batch'
# dic = {}
dic = unpickle(fil)
test_dic = unpickle(test_file)


print(dic.keys())
print(test_dic.keys())
print(len(dic[b'labels']))
print(len(test_dic[b'labels']))
print(dic[b'data'].shape)

# numbers = {}
# for i in dic[b'labels']:
#     if i in numbers.keys():
#         numbers[i] += 1
#     else:
#         numbers[i] = 1
# for i, j in numbers.items():
#     print(i, j)


x = dic[b'data']
print(x.shape)
x = np.c_[x, np.ones(x.shape[0])]
# #
# for i in x:
#     print(i)

lis = np.array([[0,1,2,5],[0,1,2,88],[4,6,8,0]])
# lis[np.max()]
print(lis.shape)
print(np.argmax(lis, axis=1))
max1 = np.argmax(lis, axis=1)
max1 = max1.reshape((max1.shape[0],1))
print(max1)
print(max1.shape)
max1[max1!=3] = 1
print(max1)
