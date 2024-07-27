from math import sqrt
from numpy import concatenate
import numpy as np
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Flatten, Conv1D
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import tensorflow as tf
import pandas as pd
import sys

import keras
import numpy as np
import sklearn.metrics as sklm

import all_models

print("Please make necessary code changes as per the dataset")
# # change shape if selected feature dataset is used
# test_X=np.empty((0,53),float)
# test_y=np.empty((0,1),int)


# # choose model from all_models
# model=all_models.GRU([100,53],0.7) #(shape,dropout) in accordance to dataset

# i=0
# with open(sys.argv[1]) as f:
#     lines=f.readlines()
#     for line in lines:
#         myarray = np.fromstring(line, dtype=float, sep=',')
#         if myarray.size!=0:
#             test_y=np.array([myarray[-1]])
#             myarray=myarray[:-1]
#             test_X=np.append(test_X,[myarray],axis=0)
#             i+=1
#             if(i==100):
#                 y=model.predict(np.reshape(test_X,[1,100,53]))
#                 print(y,test_y)
#                 test_X=np.delete(test_X,0,axis=0)
#                 test_y=np.empty((0,1),int)
#                 i=99
            

# window_size=int(sys.argv[1]) #first parameter in terminal, must match with timestep parameter used in npy_generator.py
window_size=int(128) #first parameter in terminal, must match with timestep parameter used in npy_generator.py
input_x = np.load("./TRAINING_X_"+str(window_size)+".npy")
input_y = np.load("./TRAINING_Y_"+str(window_size)+".npy")
num_windows=np.size(input_x,0)
window_size=np.size(input_x,1)
num_features=np.size(input_x,2)

model = all_models.get_optfastgrnnlstm([window_size, num_features], dropout=0.1)

# NOTE end

# NOTE time to train ##################################################################################################
keras.backend.get_session().run(tf.global_variables_initializer())
import csv

# print('*******************pretrain eval*******************')
# with open ('training/evaluation_before_training_2_40features.csv', 'w') as pretrainfile:
#     writer = csv.writer(pretrainfile)
#     writer.writerow(model.evaluate(input_x, input_y))

# print('*******************training*******************')
# model.fit(input_x, input_y)
# print("*TRAINING DONE*")
# print('*******************save model*******************')
# model.save("training/pi_queue_model_26features_5epochs.h5")
# print("*MODEL SAVED*")
# print('*******************save weights*******************')
# model.save_weights("training/pi_queue_model_weights_26features_5epochs.tf")
# print("*WEIGHTS SAVED*")

# print('*******************trained eval*******************')
# with open ('training/evaluation_after_training_2_40features.csv', 'w') as posttrainfile:
#     writer = csv.writer(posttrainfile)
#     writer.writerow(model.evaluate(input_x, input_y))

# NOTE end train ######################################################################################################

print('*******************evaluate eval*******************')
# model.load_weights("training/pi_queue_model_weights_26features_5epochs.tf") # Note: this doesnt load the weights and accuracy is like the model has not been trained yet
model = keras.models.load_model(filepath="training/pi_queue_model_26features_5epochs.h5") # Note AttributeError: 'str' object has no attribute 'decode'

test_x = np.load("./TESTING_X_"+str(window_size)+".npy")
test_y = np.load("./TESTING_Y_"+str(window_size)+".npy")
model.evaluate(input_x,input_y)
model.evaluate(test_x,test_y)



#NOTE File Shapes
# using pandas DataFrame #######################
# modified_dataset.csv =(225744, 80) (did not use header=None)
# training.csv = (158021, 80) (did not use header=None)
# testing.csv = (67722, 80) (did not use header=None)
# using Numpy ##################################
# modified_dataset.csv = (225745, 80)
# training.csv = (158022, 80)
# testing.csv = (67723, 80)
# training_x_5.npy = (158013, 5, 79)
# training_y_5.npy = (158013,)
# testing_x_5.npy = (67716, 5, 79)
# testing_y_5.npy = (67716,)

#NOTE training.csv row 6797, 14740, 15048 column 14 / column P is empty, temporary placed 0
#NOTE testing.csv row 51707(209728) column 14 / column P is empty, temporary placed 0
#NOTE the original values of above empty entries are 'NaN' in the raw dataset
#NOTE Other 'Infinity' entries in column 15 / column Q also have 'Infinity' in their column 14 / column P