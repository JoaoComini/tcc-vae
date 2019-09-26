import pandas as pd
import numpy as np

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

replace_map = {
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
    'back': 'DoS',
    'xterm': 'U2R',
    'ps' : 'U2R',
    'xlock' : 'R2L',
    'xsnoop' : 'R2L',
    'worm' : 'DoS',
    'udpstorm' : 'DoS',
    'sqlattack' : 'U2R',
    'snmpguess' : 'R2L',
    'snmpgetattack' : 'R2L',
    'sendmail' : 'R2L',
    'saint' : 'Probe',
    'processtable' : 'DoS',
    'named' : 'R2L',
    'mscan' : 'Probe',
    'httptunnel' : 'R2L',
    'apache2' : 'DoS',
    'mailbomb' : 'DoS'
}

_csv_train_data = pd.read_csv('dataset/KDDTrain+.txt', sep=',', header=None).drop(columns=[42])
_csv_train_data.columns = features

def get_dataset(mode=None):
    categorical = pd.get_dummies(_csv_train_data[['protocol_type', 'service', 'flag']])
    data = _csv_train_data.drop(columns=['protocol_type', 'service', 'flag'])
    data = pd.concat([categorical, data], axis=1)

    if mode == 'normal':
        data = data[data['class'] == 'normal']
    elif mode == 'anormal':
        data = data[data['class'] != 'normal']
    else:
        data = data

    x = data.drop(columns=['class']).astype('float64')
    y = data['class'].replace(replace_map)
    y = pd.Series(y, dtype='category')
    
    return x, y;