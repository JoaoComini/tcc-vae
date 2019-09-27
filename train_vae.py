import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from vae import Vae
import kdd

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-w', '--weights', help='Input file to load h5 model trained weights.')
    group.add_argument('-f', '--file', help='Input file to save trained model weights.')

    parser.add_argument('-p', '--plot', action='store_true', help='Plot the latent space in a 2D scatter (if the latent space dimesion is greater than 2, PCA will be applied).')
    args = parser.parse_args()

    x, y_true = kdd.get_dataset()
    x = MinMaxScaler().fit_transform(x)

    x_train, x_test = train_test_split(x, test_size=0.2)

    original_dim = x_train.shape[1]

    batch_size = 192
    epochs = 100

    vae = Vae([original_dim, 96, 64, 32, 16])

    if args.weights:
        vae.model.load_weights(args.weights)
    else:
        vae_history = vae.model.fit(
            x_train, x_train, 
            validation_data=(x_test, x_test),
            epochs=epochs, 
            batch_size=batch_size, 
            shuffle=True,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=3),
            ])

        plt.plot(vae_history.history['loss'])
        plt.plot(vae_history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.show()

    if args.plot:
        x_true, _ = kdd.get_dataset()

        x_true = MinMaxScaler().fit_transform(x_true)

        encoded_data, _, _ = vae.encoder.predict(x_true)

        decoded_data = vae.decoder.predict(encoded_data)

        losses = K.eval(binary_crossentropy(x_true, decoded_data) * original_dim)

        if len(encoded_data[0]) != 2:
            encoded_data = PCA(n_components=2).fit_transform(encoded_data)

        n_categories = len(y_true.cat.categories)

        cmap = plt.get_cmap('viridis', n_categories)
        fig, ax = plt.subplots()
        cax = ax.scatter(encoded_data[:, 0], encoded_data[:, 1], s=losses*10, c=y_true.cat.codes, edgecolors='w', cmap=cmap, alpha=0.4)
        cbar = fig.colorbar(cax)
        tick_locs = (np.arange(n_categories) + 0.5)*(n_categories-1)/n_categories
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels(y_true.cat.categories)
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(encoded_data[:, 0], encoded_data[:, 1], losses, edgecolors='w', c=y_true.cat.codes, cmap=cmap, alpha=0.6)
        plt.show()