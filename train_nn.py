from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from vae import Vae
import kdd
from utils import result_info

import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-l', '--load', help='File to load NN trained model.')
    group.add_argument('-s', '--save', help='File to save NN trained model.')

    parser.add_argument('-e', '--encode', help='Encode the training data with a Variational Autoencoder.', action='store_true')
    args = parser.parse_args()

    x, y = kdd.get_dataset()

    y = y.cat.add_categories(['anormal'])
    y[y != 'normal'] = 'anormal'

    y = y.cat.remove_unused_categories()

    x = MinMaxScaler().fit_transform(x)
    y = LabelBinarizer().fit_transform(y)

    y = to_categorical(y)

    input_dim = x.shape[1]

    if args.encode:
        vae = Vae([input_dim, 96, 64, 32, 16])
        vae.model.load_weights('models/vae_full.h5')
        x, _, _ = vae.encoder.predict(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=x_train.shape[1]))
    model.add(Dropout(0.4))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(y.shape[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if args.save:
        vae_history = model.fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            epochs=50,
            batch_size=192,
            shuffle=True,
            callbacks=[
                ModelCheckpoint(
                    filepath=args.save,
                    save_best_only=True,
                    monitor='val_accuracy',
                    save_weights_only=True
                ),
            ]
        )

        plt.plot(vae_history.history['loss'])
        plt.plot(vae_history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.show()

        plt.plot(vae_history.history['accuracy'])
        plt.plot(vae_history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='lower right')
        plt.show()

        model.load_weights(args.save)
    
    elif args.load:
        model.load_weights(args.load)

    evaluation = model.evaluate(x_test, y_test, batch_size=192)

    result_info([np.argmax(y_pred) for y_pred in model.predict(x_test[:10])], [np.argmax(test) for test in y_test[:10]], evaluation[1], evaluation[0])

        
