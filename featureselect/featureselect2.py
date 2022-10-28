from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import numpy as np
import pandas as pd

train_x=pd.read_csv('D:/freeEdu/Edge-Detect-master/training_1.csv')
train_y=train_x.iloc[:,-1:]
train_x = train_x.iloc[:,:-1]
train_y = np.array(train_y)
print(train_x.shape)
print(train_y.shape)
labels = pd.read_csv('D:/freeEdu/Edge-Detect-master/T_Headers_1.csv')

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

sfs1 = SFS(knn,k_features=16,forward=True,floating=False,verbose=2,scoring='accuracy',cv=2,n_jobs=3,fixed_features=(0,14,15,16,17,22,23,24,26,27,28,31,20,40,1,72))
# ,fixed_features=(0,0.14,0.15,0.16,0.17,0.22,0.23,0.24,0.26,0.27,0.28,0.31,20))

# (sfs13.k_feature_idx_)
# (1, 6, 14, 15, 19, 20, 21, 23, 24, 25, 40, 72, 75)
# (sfs25.k_feature_idx_)
# (1, 6, 14, 15, 19, 20, 21, 23, 24, 25, 40, 63, 65, 72, 75, 26, 70, 3, 4, 7, 5, 61, 8, 9, 11)
# array([ 1,  3,  4,  5,  6,  7,  8,  9, 11, 14, 15, 19, 20, 21, 23, 24, 25,
#        26, 40, 61, 63, 65, 70, 72, 75])

# sfs25 = SFS(knn,k_features=25,forward=True,floating=False,verbose=2,scoring='accuracy',cv=2,n_jobs=3,fixed_features=(1, 6, 14, 15, 19, 20, 21, 23,
#                                                                                                                       24, 25, 40, 63, 65, 72, 75, 26,
#                                                                                                                       70, 3, 4, 7, 5, 61))

# sfs25 = sfs25.fit(train_x, train_y.ravel())

# print(sfs25.k_feature_idx_)

# (1, 6, 14, 15, 19, 20, 21, 23, 24, 25, 40, 63, 65, 72, 75, 26, 70, 3, 4, 7, 5, 61, 8, 9, 11)
# sfs1.fit()
# sfs1.finalize_fit()
# sfs1.fit_transform()
# sfs1.transform()
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