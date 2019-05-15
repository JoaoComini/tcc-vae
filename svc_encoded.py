import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from joblib import dump, load

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.metrics import accuracy_score

data = pd.read_csv('dataset/KDDTrain+.txt', sep=',', header=None)
categorical = data[[1, 2, 3]]
categorical_dummies = pd.get_dummies(categorical)
data = data.drop(columns=[1, 2, 3])
data_encoded = pd.concat([categorical_dummies, data], axis=1)
x_train = data_encoded.drop(columns=[41, 42]).astype('float32')
y_train = data_encoded[41].values

x_train = (x_train - x_train.min())/(x_train.max() - x_train.min())
x_train = x_train.fillna(0)

encoder = load_model('models/ae_encoder.meta')
encoded_data = encoder.predict(x_train, batch_size=10)

print(encoded_data)
exit()

clf = svm.SVC(gamma='scale')
score = clf.fit(encoded_data, y_train).score(encoded_data, y_train)

print("Accuracy: {0}%\n".format(score*100))

dump(clf, 'models/svc.meta')