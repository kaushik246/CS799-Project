import os
import re
import csv
import math
from collections import defaultdict
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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

    def get_fall_params(self):
        count = len(self.parsed_data)
        acc_arr = []
        acc_x = []
        acc_y = []
        acc_z = []
        for i in range(count):
            data_point = defaultdict()
            data_point['svm'] = math.sqrt(float(self.parsed_data[i]['BELT_ACC_X'])**2 +
                                          float(self.parsed_data[i]['BELT_ACC_Y'])**2 +
                                          float(self.parsed_data[i]['BELT_ACC_Z'])**2)
            acc_x.append(float(self.parsed_data[i]['BELT_ACC_X']))
            acc_y.append(float(self.parsed_data[i]['BELT_ACC_Y']))
            acc_z.append(float(self.parsed_data[i]['BELT_ACC_Z']))
            acc_arr.append(data_point['svm'])
        plt.plot(acc_arr)
        plt.plot(acc_x)
        plt.plot(acc_y)
        plt.plot(acc_z)
        plt.show()

        #for sensor_data in self.parsed_data:
        #    data_point = defaultdict()
        #    data_point['svm'] = math.sqrt(sensor_data['BELT_ACC_X']**2, sensor_data['BELT_ACC_Y']**2,
        #                                   sensor_data['BELT_ACC_Z']**2)
        #    data_point['theta'] = math.atan(math.sqrt(sensor_data['BELT_ACC_Y']**2+sensor_data['BELT_ACC_Z']**2)/sensor_data['BELT_ACC_X'])*(180/math.pi)
        #   data_point['dsvm'] =

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
                    self.parsed_data.append(sensor_data)
        except FileNotFoundError:
            return False
        acc_x_data, acc_y_data, acc_z_data = [], [], []
        gyr_x_data, gyr_y_data, gyr_z_data = [], [], []
        acc_data, gyr_data = pd.DataFrame(), pd.DataFrame()
        for sample in self.parsed_data:
            acc_x_data.append(sample['acc_x_1'])
            acc_y_data.append(sample['acc_y_1'])
            acc_z_data.append(sample['acc_z_1'])
            gyr_x_data.append(sample['rot_x'])
            gyr_y_data.append(sample['rot_y'])
            gyr_z_data.append(sample['rot_z'])
        #plt.plot(acc_x_data)
        #plt.plot(acc_y_data)
        #plt.plot(acc_z_data)
        acc_data['fx'] = pd.Series(self.butterworth_low_pass(acc_x_data, 5.0, 200.0, 4))
        acc_data['fy'] = pd.Series(self.butterworth_low_pass(acc_y_data, 5.0, 200.0, 4))
        acc_data['fz'] = pd.Series(self.butterworth_low_pass(acc_z_data, 5.0, 200.0, 4))
        acc_data['bx'] = acc_data['fx'].diff()*(200)
        acc_data['by'] = acc_data['fy'].diff()*(200)
        acc_data['bz'] = acc_data['fz'].diff()*(200)
        acc_data['bx_2'] = acc_data['bx']**2
        acc_data['by_2'] = acc_data['by']**2
        acc_data['bz_2'] = acc_data['bz']**2

        gyr_data['rot_x'] = pd.Series(gyr_x_data)**2
        gyr_data['rot_y'] = pd.Series(gyr_y_data)**2
        gyr_data['rot_z'] = pd.Series(gyr_z_data)**2

        j1 = acc_data['bx_2'] + acc_data['by_2'] + acc_data['bz_2']
        j2 = gyr_data['rot_x'] + gyr_data['rot_y'] + gyr_data['rot_z']
        #plt.plot(j1)
        #plt.show()

        plt.plot(j2)
        plt.show()
        #print(acc_data)
        #plt.plot(acc_data['bx'])
        #plt.plot(acc_data['by'])
        #plt.plot(acc_data['bz'])
        self.max_val = math.sqrt(j1.max())
        return True
        #plt.plot(j1)
        #plt.plot(acc_x_data)
        #plt.plot(filtered_data_z)
        #plt.show()
        #print(filtered_data)
        #plt.show()

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
    '''
    def kf_predict(self, X, P, A, Q, B, U):
        X = np.dot(A, X) + np.dot(B, U)
        P = np.dot(A, np.dot(P, np.transpose(A))) + Q
        return (X, P)

    def kf_update(self, X, P, Y, H, R):
        IM = np.dot(H, X)
        IS = R + np.dot(H, np.dot(P, np.transposeH))
        K = np.dot(P, np.dot(np.transpose(H), np.inv(IS)))
        X = X + np.dot(K, (Y-IM))
        P = P - np.dot(K, np.dot(IS, np.transpose(K)))
        LH = self.gauss_pdf(Y, IM, IS)
        return X, P, K, IM, IS, LH

    def gauss_pdf(self, X, M, S):
        if M.shape()[1] == 1:
            DX = X - tile(M, X.shape()[1])
            E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
            E = E + 0.5 * M.shape()[0] * log(2 * pi) + 0.5 * log(det(S))
            P = exp(-E)
        elif X.shape()[1] == 1:
            DX = tile(X, M.shape()[1]) - M
            E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
            E = E + 0.5 * M.shape()[0] * log(2 * pi) + 0.5 * log(det(S))
            P = exp(-E)
        else:
            DX = X - M
            E = 0.5 * dot(DX.T, dot(inv(S), DX))
            E = E + 0.5 * M.shape()[0] * log(2 * pi) + 0.5 * log(det(S))
            P = exp(-E)
        return (P[0], E[0])
    '''

fall = False
for k in range(1, 6):
    trial = '_R0' + str(k) + '.txt'
    for j in range(1, 24):
        person = 'SA'
        if j < 10:
            person += '0' + str(j)
        else:
            person += str(j)
        for i in range(1, 16):
            for type in ['D']:
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
                    if data.get_sis_fall_params():
                        if data.detection():
                            if fall:
                                true_true += 1
                            else:
                                false_true += 1
                        else:
                            if fall:
                                true_false += 1
                            else:
                                false_false += 1
                except ValueError:
                    pass

sensitivity = true_true/(true_true+true_false)
specificity = false_false/(false_false+false_true)
accuracy = (sensitivity + specificity) / 2.0
print(sensitivity, specificity, accuracy)

#data = DataManager('F08_SA01_R01.txt')
#data.get_sis_fall_params()
