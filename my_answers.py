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
    import string
# define lists of letters, number, and punctuations which are valid
    sLet = list(string.ascii_letters)
# include punctuation found in text + also identified in wikipedia as English punctuation,
# under "Frequency" section of article:  https://en.wikipedia.org/wiki/Punctuation_of_English
    sPunc = ['.', ',', ';', ':', '!', '?', "'", '"', '-']
    text_to_remove = []
# Loop over text and check for valid characters.  If invalid, place on removal list.
    for iChar in text:
        if iChar in sLet or iChar in sPunc or iChar==' ':
        # normal english character, punctuation, or space
            continue
        else:
        # some anomaly - remove
            text_to_remove.append(iChar)
        
# remove as many non-english characters and character sequences as you can 
# Print out text to remove to check what has been extracted for diagnostics
    print ('text_to_remove:')
    print (text_to_remove)
# Loop over text and substitute anomalous characters with blank spaces.
    for iChar in text_to_remove:
    # Replace anomalous character with space.
        text = text.replace(iChar,' ')
    
# shorten any extra dead space created above
    text = text.replace('  ',' ')


### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    # set a sample index counter for the return array input
    sample_index=0
    # Loop over the input text until data is exhausted, with sliding window
    while sample_index+window_size < len(text):
        # create a sample vector at each sample which covers the input window
        sample=[]
        for win_index in range(window_size):
            sample.append(str(text[sample_index+win_index]))
        
        # update windowed input
        inputs.append(sample)
        # obtain the next value beyond the window as the output for the current sample    
        outputs.append(text[sample_index+window_size])
        # advance the window by one step_size
        sample_index=sample_index+step_size
    
    # diagnostics
    print('inputs[0]:', inputs[0])
    print('outputs[0]:', outputs[0])
    print('inputs[1]:', inputs[1])
    print('outputs[1]:', outputs[1])
    
    # return results
    return inputs,outputs
