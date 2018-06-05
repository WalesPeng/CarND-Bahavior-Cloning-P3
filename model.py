import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import pandas as pd
tf.cond = tf 
import keras
from keras import optimizers, backend
from keras.models import Sequential
from keras.layers import core, convolutional, pooling
#from keras.layers import Dense, Dropout, Flatten
from keras.layers import Flatten, Dense, Lambda, Dropout
from sklearn import model_selection
from data import generate_samples, preprocess
from weights_logger_callback import WeightsLogger

local_project_path = '/home/wales/Udacity/behavioral-cloning-master-track2/behavioral-cloning-master/'
local_data_path = os.path.join(local_project_path, 'data/')
local_image_path = os.path.join(local_data_path, 'IMG/')


if __name__ == '__main__':
    # Read the data
    df = pd.io.parsers.read_csv(os.path.join(local_data_path, 'driving_log.csv'))
    
    
    # Split data into training and validation sets
    df_train, df_valid = model_selection.train_test_split(df, test_size=.2)

    # Model architecture
    model = Sequential()
    model.add(convolutional.Conv2D(16, (3, 3), input_shape=(38, 128, 3), activation='relu'))
    model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
    model.add(convolutional.Conv2D(32, (3, 3), activation='relu'))
    model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
    model.add(convolutional.Conv2D(64, (3, 3), activation='relu'))
    model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    #model.add(Dropout(0.25))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=optimizers.Adam(lr=1e-04), loss='mean_squared_error')

    # print(local_image_path)
    history = model.fit_generator(
        generate_samples(df_train, local_image_path, augment=False),
        samples_per_epoch=df_train.shape[0],
        nb_epoch=10,
        validation_data=generate_samples(df_valid, local_image_path, augment=False),
        callbacks=[WeightsLogger(root_path=local_project_path)],
        nb_val_samples = df_valid.shape[0]
    )

    with open(os.path.join(local_project_path, 'model.json'), 'w') as file:
        file.write(model.to_json())

    backend.clear_session()
