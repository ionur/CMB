import numpy as np 
from sklearn.model_selection import train_test_split

rand_seed = 19951202
np.random.seed(rand_seed)

K_QE = np.load('../data/K_QE.npy')
K = np.load('../data/K.npy')
Q = np.load('../data/Q.npy')
U = np.load('../data/U.npy')

idx = np.random.permutation(K.shape[0])
K_QE_train = K_QE[idx[:13500]]
K_train = K[idx[:13500]]
Q_train = Q[idx[:13500]]
U_train = U[idx[:13500]]

K_QE_test = K_QE[idx[13500:]]
K_test = K[idx[13500:]]
Q_test = Q[idx[13500:]]
U_test = U[idx[13500:]]

np.save('../data/K_QE_train.npy', K_QE_train)
np.save('../data/K_QE_test.npy', K_QE_test)
np.save('../data/K_train.npy', K_train)
np.save('../data/K_test.npy', K_test)
np.save('../data/Q_train.npy', Q_train)
np.save('../data/Q_test.npy', Q_test)
np.save('../data/U_train.npy', U_train)
np.save('../data/U_test.npy', U_test)
