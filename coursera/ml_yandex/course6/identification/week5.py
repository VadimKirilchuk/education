from __future__ import division, print_function
# отключим всякие предупреждения Anaconda
import warnings
warnings.filterwarnings('ignore')
import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score

# Поменяйте на свой путь к данным
PATH_TO_DATA = 'kaggle'
train_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_sessions.csv'),
                       index_col='session_id')
test_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_sessions.csv'),
                      index_col='session_id')

train_test_df = pd.concat([train_df, test_df])
train_test_df_sites = train_test_df[['site%d' % i for i in range(1, 11)]].fillna(0).astype('int')

def sparse(train_data):
    X, y = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
    indptr = np.arange(X.shape[0] + 1) * X.shape[1]
    indices = X.reshape(-1)
    data = np.ones(X.size, dtype=int)
    return csr_matrix((data, indices, indptr))[:, 1:], y

train_test_sparse, y = sparse(train_test_df_sites)
X_train_sparse = train_test_sparse[:len(train_df)]
X_test_sparse = train_test_sparse[len(train_df):]

print(train_df.shape)
print(test_df.shape)

print(X_train_sparse.shape[0], X_train_sparse.shape[1], X_test_sparse.shape[0], X_test_sparse.shape[1])

with open(os.path.join(PATH_TO_DATA, 'X_train_sparse.pkl'), 'wb') as X_train_sparse_pkl:
    pickle.dump(X_train_sparse, X_train_sparse_pkl, protocol=2)
with open(os.path.join(PATH_TO_DATA, 'X_test_sparse.pkl'), 'wb') as X_test_sparse_pkl:
    pickle.dump(X_test_sparse, X_test_sparse_pkl, protocol=2)
with open(os.path.join(PATH_TO_DATA, 'train_target.pkl'), 'wb') as train_target_pkl:
    pickle.dump(y, train_target_pkl, protocol=2)

a = pickle.load(open(os.path.join(PATH_TO_DATA, 'X_train_sparse.pkl'))

b = pickle.load(open(os.path.join(PATH_TO_DATA, 'X_test_sparse.pkl'), 'rb'))

c = pickle.load(open(os.path.join(PATH_TO_DATA, 'train_target.pkl'), 'rb'))

train_share = int(.1 * X_train_sparse.shape[0])
X_train, y_train = X_train_sparse[:train_share, :], y[:train_share]
X_valid, y_valid  = X_train_sparse[train_share:, :], y[train_share:]

print(train_share, X_train_sparse.shape)

sgd_logit = SGDClassifier(loss='log', random_state=17, n_jobs=3)
sgd_logit.fit(X_train, y_train)

print(X_train.shape)
print(X_valid.shape)
logit_valid_pred_proba = sgd_logit.predict_proba(X_valid)