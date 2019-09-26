import kdd
from vae import Vae

from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load
import argparse
import numpy as np

from utils import result_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-l', '--load', help='File to load SVC trained model.')
    group.add_argument('-s', '--save', help='File to save SVC trained model.')

    parser.add_argument('-e', '--encode', help='Encode the training data with a Variational Autoencoder.', action='store_true')
    args = parser.parse_args()

    x, y = kdd.get_dataset()

    y = y.cat.add_categories(['anormal'])
    y[y != 'normal'] = 'anormal'

    y = y.cat.remove_unused_categories()

    x = MinMaxScaler().fit_transform(x)

    if args.encode:
        vae = Vae([x.shape[1], 96, 64, 32, 16])
        vae.model.load_weights('models/vae_full.h5')
        x, _, _ = vae.encoder.predict(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
    
    if args.load:
        clf = load(args.load)
    else:
        clf = SVC(gamma='scale')
        clf.fit(x_train, y_train)
        dump(clf, args.save)
    
    y_pred = clf.predict(x_test)

    score = accuracy_score(y_test, y_pred)

    result_info(y_pred[: 10], y_test[:10].to_numpy(), accuracy=score)


