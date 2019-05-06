from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import backend as K

import numpy as np

class LSTM():
    def __init__(self):

        self.sequence_length = 17  # number of days before predicting
        self.n_features = 28  # number of features in the data set
        self.time_step = 1  # number of samples

        model = Sequential()

        model.add(LSTM(500, input_shape=(self.sequence_length, self.time_step, self.n_features),
                       kernel_initializer='glorot_uniform', stateful=True))
        model.add(Dense(1))

        model.compile(loss='smooth_L1_loss', optimizer='adam')

    def fit(self, X_train, y_train, batch_size, nb_epoch):
        for i in range(nb_epoch):
            self.model.fit(X_train, y_train, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
            self.model.reset_states()
        return model

    def predict(self, batch_size, X_test):
        return self.model.predict(X_test, batch_size=batch_size)

    def smooth_L1_loss(y_true, y_pred):
        THRESHOLD = K.variable(1.0)
        mae = K.abs(y_true - y_pred)
        flag = K.greater(mae, THRESHOLD)
        loss = K.mean(K.switch(flag, (mae - 0.5), K.pow(mae, 2)), axis=-1)
        return loss
