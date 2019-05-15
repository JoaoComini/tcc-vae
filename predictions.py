import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from joblib import dump, load
import numpy as np
import pandas as pd

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

encoder = load_model('models/vae_encoder.meta')
encoded_data = encoder.predict(x_train, batch_size=10)

clf = load('models/svc.meta')
y_pred = clf.predict(encoded_data)

print("Accuracy: {0}%".format(accuracy_score(y_train, y_pred)*100))