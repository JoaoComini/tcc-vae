import tensorflow as tf
from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

features = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "class"
]

data = pd.read_csv('dataset/KDDTrain+.txt', sep=',', header=None).drop(columns=[42])
data.columns = features

categorical = pd.get_dummies(data[["protocol_type", "service", "flag"]])
data.drop(columns=["protocol_type", "service", "flag"], inplace=True)
data = pd.concat([categorical, data], axis=1)

data['class'][data['class'] != 'normal'] = 'anormal'

full_data = data
normal_data = data[data['class'] == 'normal']
anormal_data = data[data['class'] != 'normal']

normal_data.drop(columns=["class"], inplace=True)
anormal_data.drop(columns=["class"], inplace=True)

x_train = MinMaxScaler().fit_transform(normal_data)
x_test = MinMaxScaler().fit_transform(full_data.drop(columns=["class"]))
y_test = full_data['class']

original_dim = x_train.shape[1]

input_shape = (original_dim, )
batch_size = 10
latent_dim = 10
epochs = 3

inputs = Input(shape=input_shape, name='encoder_input')
h1 = Dense(40, activation='relu')(inputs)

z_mean = Dense(latent_dim, name='z_mean')(h1)
z_log_var = Dense(latent_dim, name='z_log_var')(h1)
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

h2 = Dense(40, activation='relu')(z)
outputs = Dense(original_dim, activation='sigmoid')(h2)

encoder = Model(inputs, z_mean)
vae = Model(inputs, outputs, name='ae_mlp')

reconstruction_loss = mse(inputs, outputs)
kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()

vae.fit(x_train, epochs=epochs, batch_size=batch_size, shuffle=True)

predictions = []
for i in range(len(x_test)):
    predictions.append('normal' if vae.evaluate(x_test[i:i+1], verbose=0) < 0.04 else 'anormal')

score = accuracy_score(y_test, predictions)

print("Accuracy score: {0}".format(score))


# x_test_encoded = encoder.predict(x_train, batch_size=batch_size)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], x_test_encoded[:, 2], c=y_train.cat.codes)
# plt.show()