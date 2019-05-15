import tensorflow as tf
from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('dataset/KDDTrain+.txt', sep=',', header=None)
categorical = data[[1, 2, 3]]
categorical_dummies = pd.get_dummies(categorical)

data = data.drop(columns=[1, 2, 3, 41, 42])
data_encoded = (pd.concat([categorical_dummies, data], axis=1)).astype('float32')

corr = data_encoded.corr()

columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            columns[j] = False

selected_columns = data_encoded.columns[columns]
selected_data = data_encoded[selected_columns]

x_train = MinMaxScaler().fit_transform(selected_data)

original_dim = x_train.shape[1]

input_shape = (original_dim, )
batch_size = 10
latent_dim = 10
epochs = 5

inputs = Input(shape=input_shape, name='encoder_input')
h1 = Dense(50, activation='relu')(inputs)

z = Dense(latent_dim, activation='relu')(h1)

h2 = Dense(50, activation='relu')(z)
outputs = Dense(original_dim, activation='sigmoid')(h2)

encoder = Model(inputs, z)
ae = Model(inputs, outputs, name='ae_mlp')

ae.compile(optimizer='adam', loss='mse')
ae.summary()

ae.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, shuffle=True)

ae.save('models/ae.meta')
encoder.save('models/ae_encoder.meta')
