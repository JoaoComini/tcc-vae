import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from joblib import dump, load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler, MinMaxScaler


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

y_train = data['class'][data['class'] != 'normal']

categorical = pd.get_dummies(data[["protocol_type", "service", "flag"]])

data.drop(columns=["protocol_type", "service", "flag", "class"], inplace=True)
data = pd.concat([categorical, data], axis=1)

x_train = MinMaxScaler().fit_transform(data)

y_train.replace({
    'warezmaster' : 'R2L',
    'warezclient' : 'R2L',
    'teardrop' : 'DoS',
    'spy' : 'R2L',
    'smurf' : 'DoS',
    'satan' : 'Probe',
    'rootkit' : 'U2R',
    'portsweep' : 'Probe',
    'pod' : 'DoS',
    'phf' : 'R2L',
    'perl' : 'U2R',
    'nmap' : 'Probe',
    'neptune' : 'DoS',
    'multihop' : 'R2L',
    'loadmodule' : 'U2R',
    'land' : 'DoS',
    'ipsweep' : 'Probe',
    'imap' : 'R2L',
    'guess_passwd' : 'R2L',
    'ftp_write' : 'R2L',
    'buffer_overflow' : 'U2R',
    'back': 'DoS'
}, inplace=True)

y_train = pd.Series(y_train, dtype="category")

n_categories = len(y_train.cat.categories)

vae = load_model('models/vae.meta')
score = vae.evaluate(x_train, batch_size=10)

print(score)
exit()

transformed = PCA(n_components=2).fit_transform(encoded_data)

cmap = plt.get_cmap('viridis', n_categories)

fig, ax = plt.subplots()
cax = ax.scatter(transformed[:, 0], transformed[:, 1], c=y_train.cat.codes, cmap=cmap)
cbar = fig.colorbar(cax)
tick_locs = (np.arange(n_categories) + 0.5)*(n_categories-1)/n_categories
cbar.set_ticks(tick_locs)
cbar.set_ticklabels(y_train.cat.categories)
plt.show()