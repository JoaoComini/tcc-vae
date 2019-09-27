import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
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

    x, _ = kdd.get_dataset(mode='normal')

    x = MinMaxScaler().fit_transform(x)

    x_train, x_test = train_test_split(x, test_size=0.2)

    original_dim = x_train.shape[1]

    batch_size = 48
    epochs = 100

    vae = Vae([original_dim, 64, 32, 16, 8])

    if args.file:
        vae_history = vae.model.fit(
            x_train, x_train,
            validation_data=(x_test, x_test),
            epochs=epochs, 
            batch_size=batch_size, 
            shuffle=True,
            callbacks=[
                ModelCheckpoint(
                    filepath=args.file,
                    save_weights_only=True,
                    save_best_only=True,
                ),
                EarlyStopping(
                    patience=3
                )
            ])

        plt.plot(vae_history.history['loss'], label='Train')
        plt.plot(vae_history.history['val_loss'], label='Test')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.show()
    
    vae.model.load_weights(args.weights if args.weights else args.file)

    x_true, y = kdd.get_dataset()

    y_true = y.cat.add_categories(['anormal'])
    y_true[y_true != 'normal'] = 'anormal'

    y_true = y_true.cat.remove_unused_categories()

    x_true = MinMaxScaler().fit_transform(x_true)

    x_pred = vae.model.predict(x_true)

    losses = K.eval(vae.loss(x_true, x_pred) * original_dim)

    y_pred = ['normal' if loss <= 7 else 'anormal' for loss in losses]

    print(accuracy_score(y_true, y_pred))

    if args.plot:
        x_encoded, _, _ = vae.encoder.predict(x_true)

        if x_encoded.shape[1] != 2:
            x_encoded = PCA(n_components=2).fit_transform(x_encoded)

        n_categories = len(y.cat.categories)

        cmap = plt.get_cmap('viridis', n_categories)
        fig, ax = plt.subplots()
        fig.suptitle('Latent Space')
        cax = ax.scatter(x_encoded[:, 0], x_encoded[:, 1], s=losses*10, c=y.cat.codes, edgecolors='w', cmap=cmap, alpha=0.4)
        cbar = fig.colorbar(cax)
        tick_locs = (np.arange(n_categories) + 0.5)*(n_categories-1)/n_categories
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels(y.cat.categories)
        plt.show()

        fig = plt.figure()
        fig.suptitle('Latent Space 3D')
        ax = fig.add_subplot(111, projection='3d')
        cax = ax.scatter(x_encoded[:, 0], x_encoded[:, 1], losses, edgecolors='w', c=y.cat.codes, cmap=cmap, alpha=0.6)
        cbar = fig.colorbar(cax)
        tick_locs = (np.arange(n_categories) + 0.5)*(n_categories-1)/n_categories
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels(y.cat.categories)
        plt.show()

