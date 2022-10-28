import sys

import numpy
import tensorflow as tf             #NOTE additional import for initialization line 21
from tensorflow.python import keras #NOTE additional import for initialization line 21
# keras.backend.get_session().run(tf.global_variables_initializer()) #NOTE
# keras.backend.get_session().run(tf.initialize_all_variables())    #NOTE

import numpy as np

import all_models  # NOTE copied all_models.get_optfastgrnnlstm_single_layer from testing folder

print("Please make necessary code changes as per the dataset")
# change shape if selected feature dataset is used
# test_X=np.empty((0,53),float)
# test_X = np.empty((0, 26), float)  #NOTE 2nd parameter = number of features in ./training.csv ############################################### 53->79
# test_y = np.empty((0, 1), int)
#
# # choose model from all_models
# model = all_models.get_optfastgrnnlstm([100, 26], dropout=0.1)  #NOTE (shape,dropout) in accordance to dataset ####################################### 53->79
# keras.backend.get_session().run(tf.global_variables_initializer())  # NOTE fix for uninitialized variable error
#
# i = 0
# import csv
# # training_predictions = open ('training/training predictions 9 26features_5epochs.csv', 'w')
# training_predictions = open ('testing/testing predictions 9 26features_5epochs.csv', 'w')
# writer = csv.writer(training_predictions)
# printrow = [['predicted y', 'actual y']]
# model.load_weights('training/pi_queue_model_weights_26features_5epochs.tf')
# with open(sys.argv[1]) as f:
#     counter = 1 #NOTE debug/monitoring
#     lines = f.readlines()
#     for line in lines:
#         myarray = np.fromstring(line, dtype=float, sep=',')
#         if myarray.size != 0:
#             test_y = np.array([myarray[-1]])
#             myarray = myarray[:-1]
#
#             test_X = np.append(test_X, [myarray], axis=0)
#             i += 1
#             if (i == 100):
#                 # model.fit(np.reshape(test_X, [1, 100, 40]),test_y)
#                 y = model.predict(np.reshape(test_X, [1, 100, 26]))  #NOTE 3rd parameter = number of features
#                 print(y, test_y, counter) #NOTE added test_y/y counter for debug/monitoring
#                 printrow.append([y,test_y])
#                 test_X = np.delete(test_X, 0, axis=0)
#                 test_y = np.empty((0, 1), int)
#                 i = 99
#                 counter += 1 #NOTE debug/monitoring
#
# writer.writerows(printrow)
#
# # model.save_weights('training/training_model_weights_4.h5py',save_format='h5',overwrite=False)


# NOTE

window_size=int(sys.argv[1]) #first parameter in terminal, must match with timestep parameter used in npy_generator.py
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