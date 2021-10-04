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

# actual_detection
true_true = 0
true_false = 0
false_true = 0
false_false = 0

class DataManager:
    def __init__(self, filename, index, person, fall):
        self.filename = filename
        self.index = index
        self.person = person
        self.fall = fall
        self.parsed_data = []
        self.params = []
        self.threshold = 600
        self.acc_1_factor = (2*16)/(2**13)
        self.gyro_factor = (2*2000)/(2**16)
        self.acc_2_factor = (2*8)/(2**14)
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

    def get_sis_fall_params(self):
        try:
            with open('/Users/kaushikkota/ms_cs_uw_madison/cs_799/CS799-Project/SisFall_dataset/' + self.person + '/' + self.filename) as f:
                contents = f.readlines()
                for line in contents:
                    sensor_data = {}
                    data_sample = line.strip().replace(';', '').split(',')
                    for i in range(9):
                        formatted_sample = data_sample[i].lstrip()
                        sensor_data[self.sis_params[i]] = int(formatted_sample)
                        if i in [0, 1, 2]:
                            sensor_data[self.sis_params[i]] = self.acc_1_factor*sensor_data[self.sis_params[i]]
                        elif i in [3, 4, 5]:
                            sensor_data[self.sis_params[i]] = self.gyro_factor*sensor_data[self.sis_params[i]]
                        elif i in [6, 7, 8]:
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

        acc_data['fx'] = pd.Series(self.butterworth_low_pass(acc_x_data, 5.0, 200.0, 4)).dropna()
        acc_data['fy'] = pd.Series(self.butterworth_low_pass(acc_y_data, 5.0, 200.0, 4)).dropna()
        acc_data['fz'] = pd.Series(self.butterworth_low_pass(acc_z_data, 5.0, 200.0, 4)).dropna()

        gyr_data['fx'] = pd.Series(gyr_x_data).dropna()
        gyr_data['fy'] = pd.Series(gyr_y_data).dropna()
        gyr_data['fz'] = pd.Series(gyr_z_data).dropna()

        acc2_data['fx'] = pd.Series(self.butterworth_low_pass(acc2_x_data, 5.0, 200.0, 4)).dropna()
        acc2_data['fy'] = pd.Series(self.butterworth_low_pass(acc2_y_data, 5.0, 200.0, 4)).dropna()
        acc2_data['fz'] = pd.Series(self.butterworth_low_pass(acc2_z_data, 5.0, 200.0, 4)).dropna()

        feature_dict = {}

        # Accelerometer 1 Data
        feature_dict['max_amp_x'] = np.max(acc_data['fx'])
        feature_dict['min_amp_x'] = np.min(acc_data['fx'])
        feature_dict['mean_amp_x'] = np.mean(acc_data['fx'])
        feature_dict['variance_x'] = np.var(acc_data['fx'])
        feature_dict['kurtosis_x'] = self.get_kurtosis(acc_data['fx'])
        feature_dict['skew_x'] = self.get_skew(acc_data['fx'])

        feature_dict['max_amp_y'] = np.max(acc_data['fy'])
        feature_dict['min_amp_y'] = np.min(acc_data['fy'])
        feature_dict['mean_amp_y'] = np.mean(acc_data['fy'])
        feature_dict['variance_y'] = np.var(acc_data['fy'])
        feature_dict['kurtosis_y'] = self.get_kurtosis(acc_data['fy'])
        feature_dict['skew_y'] = self.get_skew(acc_data['fy'])

        feature_dict['max_amp_z'] = np.max(acc_data['fz'])
        feature_dict['min_amp_z'] = np.min(acc_data['fz'])
        feature_dict['mean_amp_z'] = np.mean(acc_data['fz'])
        feature_dict['variance_z'] = np.var(acc_data['fz'])
        feature_dict['kurtosis_z'] = self.get_kurtosis(acc_data['fz'])
        feature_dict['skew_z'] = self.get_skew(acc_data['fz'])

        # Accelerometer 2 Data
        feature_dict['max_amp_x_2'] = np.max(acc2_data['fx'])
        feature_dict['min_amp_x_2'] = np.min(acc2_data['fx'])
        feature_dict['mean_amp_x_2'] = np.mean(acc2_data['fx'])
        feature_dict['variance_x_2'] = np.var(acc2_data['fx'])
        feature_dict['kurtosis_x_2'] = self.get_kurtosis(acc2_data['fx'])
        feature_dict['skew_x_2'] = self.get_skew(acc2_data['fx'])

        feature_dict['max_amp_y_2'] = np.max(acc2_data['fy'])
        feature_dict['min_amp_y_2'] = np.min(acc2_data['fy'])
        feature_dict['mean_amp_y_2'] = np.mean(acc2_data['fy'])
        feature_dict['variance_y_2'] = np.var(acc2_data['fy'])
        feature_dict['kurtosis_y_2'] = self.get_kurtosis(acc2_data['fy'])
        feature_dict['skew_y_2'] = self.get_skew(acc2_data['fy'])

        feature_dict['max_amp_z_2'] = np.max(acc2_data['fz'])
        feature_dict['min_amp_z_2'] = np.min(acc2_data['fz'])
        feature_dict['mean_amp_z_2'] = np.mean(acc2_data['fz'])
        feature_dict['variance_z_2'] = np.var(acc2_data['fz'])
        feature_dict['kurtosis_z_2'] = self.get_kurtosis(acc2_data['fz'])
        feature_dict['skew_z_2'] = self.get_skew(acc2_data['fz'])

        # Gyro Data
        feature_dict['max_rot_x'] = np.max(gyr_data['fx'])
        feature_dict['min_rot_x'] = np.min(gyr_data['fx'])
        feature_dict['mean_rot_x'] = np. mean(gyr_data['fx'])
        feature_dict['variance_rot_x'] = np.var(gyr_data['fx'])
        feature_dict['kurtosis_rot_x'] = self.get_kurtosis(gyr_data['fx'])
        feature_dict['skew_x'] = self.get_skew(gyr_data['fx'])

        feature_dict['max_rot_y'] = np.max(gyr_data['fy'])
        feature_dict['min_rot_y'] = np.min(gyr_data['fy'])
        feature_dict['mean_rot_y'] = np.mean(gyr_data['fy'])
        feature_dict['variance_rot_y'] = np.var(gyr_data['fy'])
        feature_dict['kurtosis_rot_y'] = self.get_kurtosis(gyr_data['fy'])
        feature_dict['skew_rot_y'] = self.get_skew(gyr_data['fy'])

        feature_dict['max_rot_z'] = np.max(gyr_data['fz'])
        feature_dict['min_rot_z'] = np.min(gyr_data['fz'])
        feature_dict['mean_rot_z'] = np.mean(gyr_data['fz'])
        feature_dict['variance_rot_z'] = np.var(gyr_data['fz'])
        feature_dict['kurtosis_rot_z'] = self.get_kurtosis(gyr_data['fz'])
        feature_dict['skew_rot_z'] = self.get_skew(gyr_data['fz'])

        feature_dict['result'] = 1 if self.fall else 0
        return feature_dict, True

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
    for j in range(1, 24):
        persons = ['SA', 'SE']
        for person in persons:
            if j < 10:
                person += '0' + str(j)
            else:
                person += str(j)
            for i in range(1, 18):
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
y = np.array(features_mat['result'])
#print(len(y))

x = features_mat.drop(columns='result').to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True)

from sklearn.neighbors import KNeighborsClassifier


for n_neighbor in [5, 7, 10, 12, 15]:
    knn_cv = KNeighborsClassifier(n_neighbors=n_neighbor)
    cv_scores = cross_val_score(knn_cv, x, y, cv=10, n_jobs=-1)
    print("KNN", mean(cv_scores), n_neighbor)


#cv = KFold(n_splits=10, random_state=1, shuffle=True)
#classifier = svm.SVC(kernel='poly', degree=3)
#scores = cross_val_score(classifier, x, y, scoring='accuracy', cv=cv, n_jobs=1)
#print("SVM", mean(scores))

'''
match = 0
mismatch = 0
for i in range(len(y_test)):
    if cl_out[i] == y_test[i]:
        match += 1
    else:
        mismatch += 1

print(match, mismatch)



#sensitivity = true_true/(true_true+true_false)
#specificity = false_false/(false_false+false_true)
#accuracy = (sensitivity + specificity) / 2.0
#print(sensitivity, specificity, accuracy)

#data = DataManager('F08_SA01_R01.txt')
#data.get_sis_fall_params()
'''
