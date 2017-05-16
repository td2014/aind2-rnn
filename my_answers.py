import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    # containers for input/output pairs
    X = []
    y = []
    
    # set a sample index counter for the return array X
    sample_index=0
    # Loop over the input series until data is exhausted, with sliding window
    while sample_index+window_size < len(series):
        # create a sample vector at each sample which covers the input window
        sample=[]
        sample[0:window_size] = \
           series[sample_index:sample_index+window_size]
        
        # update windowed input
        X.append(sample)
        # obtain the next value beyond the window as the output for the current sample    
        y.append(series[sample_index+window_size])
        # advance the window by one time step
        sample_index=sample_index+1
        
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)
    
    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
#
# Using specifications from Jupyter notebook instructions.
# Don't know what to do with step_size here.  Ignoring for now.
#
    model = Sequential()
# Add recurrant module with 5 hidden units
    model.add(LSTM(5, input_shape=(1,window_size)))
# Add fully connected module for output - 1 output unit
    model.add(Dense(1))
# Verify architecture
    model.summary()

# build model using keras documentation recommended optimizer initialization
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)
# return model and exit
    return model


### TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    # find all unique characters in the text


    # remove as many non-english characters and character sequences as you can 


### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    

    
    return inputs,outputs
