import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
import pandas as pd
import numpy as np

class RunLSTM(object):
  def __init__(self, train, validation = [], test = []):
    self.train = train;
    self.test = test;
    self.validation = validation;
    self.maxLength=501
    self.validate_from_train = False
    if not self.validation:
      self.validate_from_train = True

    self.model = None


  def prepare_data(self, data_df):
    data = data_df.to_numpy()
    for i in range(len(data)):   
        tempdata = np.asarray([data[i][j][z] for z in range(501) for j in range(9)]).astype('float32')
        # tempdata = np.asarray(acc_x_1).astype('float32')
        tempdata = tempdata.reshape(self.maxLength, 9, order ='F')
        parsed_data.append(tempdata)
        y_data.append(data[i][9])

    parsed_data_array = np.array(parsed_data)
    y_data_array = np.asarray(y_data).astype('float32')
    return (parsed_data_array, y_data_array)
  
  def train_lstm(self):
    parsed_data = []
    y_data = []
    counter=0
    
    parsed_data_array , y_data_array = self.prepare_data(self.train)
    parsed_data_array_val = parsed_data_array
    y_data_array_val = y_data_array
    if not self.validate_from_train:
      parsed_data_array_val , y_data_array_val = self.prepare_data(self.validation)

    model = Sequential()
    model.add(LSTM(64, input_shape=(self.maxLength, 9)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(parsed_data_array, y_data_array, validation_data=(parsed_data_array_val, y_data_array_val), epochs=10, batch_size=500)
    self.model = model

  def run_lstm(self,test = []):
    if test:
      self.test = test
    parsed_data_array , y_data_array = self.prepare_data(self.test)
    predictions = self.model.predict(parsed_data_array)
    return predictions

