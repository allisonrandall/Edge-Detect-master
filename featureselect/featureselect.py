from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import numpy as np
from training import all_models
import pandas as pd
import tensorflow as tf
from tensorflow.python import keras

# train_x=np.load("D:/freeEdu/Edge-Detect-master/TRAINING_X_1_80features.npy")
# train_y=np.load("D:/freeEdu/Edge-Detect-master/TRAINING_Y_1_80features.npy")
train_x=pd.read_csv('D:/freeEdu/Edge-Detect-master/training_1.csv')
train_y=train_x.iloc[:,-1:]
train_x = train_x.iloc[:,:-1]
train_y = np.array(train_y)
print(train_x.shape)
print(train_y.shape)
labels = pd.read_csv('D:/freeEdu/Edge-Detect-master/T_Headers_1.csv')

# num_windows=np.size(train_x,0)
# window_size=np.size(train_x,1)
# num_features=np.size(train_x,2)
#
# model = all_models.get_optfastgrnnlstm([window_size, num_features], dropout=0.1)
# keras.backend.get_session().run(tf.global_variables_initializer())

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=4)

sfs1 = SFS(knn,
           k_features=25,
           forward=True,
           floating=False,
           verbose=2,
           scoring='accuracy',
           cv=0)

sfs1 = sfs1.fit(train_x, train_y.ravel(), 'custom_feature_names=labels')
print(sfs1.k_feature_names_)
print(labels)
#
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.datasets import load_iris
#
# iris = load_iris()
# X = iris.data
# y = iris.target
# knn = KNeighborsClassifier(n_neighbors=4)
#
# from mlxtend.feature_selection import SequentialFeatureSelector as SFS
#
# sfs1 = SFS(knn,
#            k_features=3,
#            forward=True,
#            floating=False,
#            verbose=2,
#            scoring='accuracy',
#            cv=0)
#
# sfs1 = sfs1.fit(X, y)
#
# print(sfs1.k_feature_names_)