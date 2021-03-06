import scipy.io as scio
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import spectral


path = './9个类别的数据集-train/'
files = os.listdir(path)
s = []

data_min = []
data_max = []
labels = []
for file in files:
    data = scio.loadmat(path+file)

    A = file[:-4]
    label = int(file[4:-10])

    amin, amax = data[A].min(axis=0), data[A].max(axis=0)
    a = (data[A]-amin)/(amax-amin)
    data_min.append(amin)
    data_max.append(amax)
    print(label, len(a))
    for i in range(len(a)):
        labels.append(label)
    s.extend(a)
train = np.array(s)

labels = np.array(labels)
test_dict = scio.loadmat('data_test_final.mat')
data = test_dict['data_test_final']
test = []
test = (data-np.array(data_min).mean(axis=0)) / \
    (np.array(data_max).mean(axis=0)-np.array(data_min).mean(axis=0))

test = np.array(test)
x_train, x_test, y_train, y_test = train_test_split(
    train, labels, test_size=0.33, random_state=0)

knn = KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree', p=1)
knn.fit(x_train, y_train)
print("acc=", knn.score(x_test, y_test))

testy = knn.predict(test)


testy = testy.tolist()
print(testy)
x = [i for i in range(len(testy))]
result = {'x': x, 'y': testy}
df = pd.DataFrame(result)
df.to_csv('testy.csv', index=False)