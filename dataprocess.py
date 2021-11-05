import os
import re
import csv
import math
from collections import defaultdict
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statistics import mean
from scipy.stats import kurtosis, skew
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
#from numpy import dot, sum, tile, linalg, det
#from numpy.linalg import inv, det
# from tensorflow import keras
 
# from keras.models import Sequential
# from keras.layers import LSTM
# from keras.layers import Dense, Dropout
# actual_detection
true_true = 0
true_false = 0
false_true = 0
false_false = 0

class DataManager:
    def __init__(self, filename, index, person, fall, align = False):
        self.filename = filename
        self.index = index
        self.person = person
        self.fall = fall
        self.parsed_data = []
        self.params = []
        self.threshold = 600
        self.acc_1_factor = (2*16)/(2**13)
        self.acc_2_factor = (2*8)/(2**14)
        self.gyr_factor = (2*2000)/2**16

        self.sis_params = {
            0: 'acc_x_1',
            1: 'acc_y_1',
            2: 'acc_z_1',
            3: 'rot_x',
            4: 'rot_y',
            5: 'rot_z',
            6: 'acc_x_2',
            7: 'acc_y_2',
            8: 'acc_z_2'
        }
        self.max_val = 0

    def read_csv_data(self):
        with open(self.filename, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                self.parsed_data.append(row)

    def align_data(self, req_len, data_seq):
      maxd = max(data_seq)
      mind = min(data_seq)
      max_point = max(maxd, -mind )
      
      max_ind = 0

      if max_point == maxd:
        max_ind = data_seq.index(maxd)
      else:
        max_ind = data_seq.index(mind)

      data_seq = [0]*(req_len//2) + data_seq + [0]*(req_len//2) 

      return data_seq[max_ind : max_ind+req_len]
      
    def get_sis_fall_params(self):
        try:
            with open('/content/drive/MyDrive/Fall-Detection/SisFall_dataset/' + self.person + '/' + self.filename) as f:
                contents = f.readlines()
                for line in contents:
                    sensor_data = {}
                    data_sample = line.strip().replace(';', '').split(',')
                    for i in range(9):
                        formatted_sample = data_sample[i].lstrip()
                        # print(formatted_sample)
                        sensor_data[self.sis_params[i]] = int(formatted_sample)
                        # print(sensor_data[self.sis_params[i]])
                        if i in [0, 1, 2]:
                            sensor_data[self.sis_params[i]] = self.acc_1_factor*sensor_data[self.sis_params[i]]
                        elif i in [3, 4, 5]:
                            sensor_data[self.sis_params[i]] = self.gyr_factor*sensor_data[self.sis_params[i]]
                        else:
                            sensor_data[self.sis_params[i]] = self.acc_2_factor*sensor_data[self.sis_params[i]]
                    self.parsed_data.append(sensor_data)

        except FileNotFoundError:
            return {}, False
        acc_x_data, acc_y_data, acc_z_data = [], [], []
        gyr_x_data, gyr_y_data, gyr_z_data = [], [], []
        acc2_x_data, acc2_y_data, acc2_z_data = [], [], []

        acc_data, gyr_data, acc2_data = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        for sample in self.parsed_data:
            acc_x_data.append(sample['acc_x_1'])
            acc_y_data.append(sample['acc_y_1'])
            acc_z_data.append(sample['acc_z_1'])
            gyr_x_data.append(sample['rot_x'])
            gyr_y_data.append(sample['rot_y'])
            gyr_z_data.append(sample['rot_z'])
            acc2_x_data.append(sample['acc_x_2'])
            acc2_y_data.append(sample['acc_y_2'])
            acc2_z_data.append(sample['acc_z_2'])
        # print(acc_x_data)
        if align:
          acc_x_data =  self.align_data(501,acc_x_data)
          acc_y_data = self.align_data(501,acc_y_data)
          acc_z_data= self.align_data(501,acc_z_data)
          gyr_x_data= self.align_data(501,gyr_x_data)
          gyr_y_data = self.align_data(501,gyr_y_data)
          gyr_z_data = self.align_data(501,gyr_z_data)
          acc2_x_data= self.align_data(501,acc2_x_data)
          acc2_y_data = self.align_data(501,acc2_y_data)
          acc2_z_data = self.align_data(501,acc2_z_data)

        feature_dict = {}
        feature_dict['afx'] = pd.Series(acc_x_data).dropna()
        feature_dict['afy'] = pd.Series(acc_y_data).dropna()
        feature_dict['afz'] = pd.Series(acc_z_data).dropna()

        feature_dict['gfx'] = pd.Series(gyr_x_data).dropna()
        feature_dict['gfy'] = pd.Series(gyr_y_data).dropna()
        feature_dict['gfz'] = pd.Series(gyr_z_data).dropna()

        feature_dict['afx2'] = pd.Series(acc2_x_data).dropna()
        feature_dict['afy2'] = pd.Series(acc2_y_data).dropna()
        feature_dict['afz2'] = pd.Series(acc2_z_data).dropna()

        feature_dict['result'] = 1 if self.fall else 0
        
        return feature_dict, True
        #plt.plot(j1)
        # plt.plot(acc_data['fx'])
        # plt.plot(acc_data['fy'])
        # plt.plot(acc_data['fz'])
        #plt.plot(filtered_data_z)
        #plt.show()
        #print(filtered_data)
        #plt.show()

    def get_kurtosis(self, data):
        return self.get_n_moment(data, 4)/(np.var(data)**2)

    def get_skew(self, data):
        return self.get_n_moment(data, 3)/(math.sqrt(np.var(data))**3)

    def get_n_moment(self, data, n):
        mean = np.mean(np.array(data))
        sum_ = 0.0
        for sample in data:
            sum_ += (sample-mean)**n
        return sum_/len(data)

    def detection(self):
        if self.max_val > self.threshold:
            return True
        else:
            return False

    def butterworth_low_pass(self, data, cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = lfilter(b, a, data)
        return y


# Start
features = []
fall = False
for k in range(1, 6):
    trial = '_R0' + str(k) + '.txt'
    for j in range(1, 26):
        persons = ['SA', 'SE']
        for person in persons:
            if j < 10:
                person += '0' + str(j)
            else:
                person += str(j)
            for i in range(1, 21):
                for type in ['F', 'D']:
                    post_fix = '_' + person + trial
                    if type == 'F':
                        fall = True
                    else:
                        fall = False
                    if i < 10:
                        filename = type + '0' + str(i) + post_fix
                    else:
                        filename = type + str(i) + post_fix
                    try:
                        data = DataManager(filename, i, person, fall)
                        feature, found = data.get_sis_fall_params()
                        if found:
                            features.append(feature)
                    except ValueError:
                        pass

features_mat = pd.DataFrame(features)
print(features_mat)
y = np.array(features_mat['result'])
print(len(y))
print("dataframe:")
print(features_mat)
