from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import numpy as np

class LSTM():
    def __init__(self):

        self.sequence_length = 17  # number of days before predicting
        self.n_features = 28  # number of features in the dataset
        self.time_step = 1  # number of samples