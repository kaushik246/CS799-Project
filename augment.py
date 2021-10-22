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

class DataManager:
    def __init__(self, filename, person, fall):
        self.filename = filename
        self.person = person
        self.fall = fall
        self.params = []
        self.parsed_data = []
        self.acc_1_factor = (2*16)/(2**13)
        self.sis_params = {
            0: 'acc_x_1',
            1: 'acc_y_1',
            2: 'acc_z_1',
        }
        self.new_path = '/Users/kaushikkota/ms_cs_uw_madison/cs_799/CS799-Project/SisFall_Phases'
        self.max_val = 0

    def get_sis_fall_params(self):
        try:
            with open('/Users/kaushikkota/ms_cs_uw_madison/cs_799/CS799-Project/SisFall_dataset/' + self.person + '/' + self.filename) as f:
                contents = f.readlines()
                for line in contents:
                    sensor_data = {}
                    data_sample = line.strip().replace(';', '').split(',')
                    for i in range(3):
                        formatted_sample = data_sample[i].lstrip()
                        sensor_data[self.sis_params[i]] = int(formatted_sample)
                        if i in [0, 1, 2]:
                            sensor_data[self.sis_params[i]] = self.acc_1_factor*sensor_data[self.sis_params[i]]
                    self.parsed_data.append(sensor_data)
        except FileNotFoundError:
            return {}, False
        acc_x_data, acc_y_data, acc_z_data, svm_data = [], [], [], []

        data_pre_peak, data_post_peak = pd.DataFrame(), pd.DataFrame()
        data_pre_impact, data_post_impact = pd.DataFrame(), pd.DataFrame()

        for sample in self.parsed_data:
            x_sample, y_sample, z_sample = sample['acc_x_1'], sample['acc_y_1'], sample['acc_z_1']
            acc_x_data.append(x_sample)
            acc_y_data.append(y_sample)
            acc_z_data.append(z_sample)
            svm_data.append(math.sqrt(x_sample*x_sample+y_sample*y_sample+z_sample*z_sample))

        acc_x_data = np.array(acc_x_data)
        acc_y_data = np.array(acc_y_data)
        acc_z_data = np.array(acc_z_data)

        peak_index = np.argmax(svm_data)
        sample_id = self.filename.split('.')[0]
        os.mkdir(self.new_path + '/' + sample_id)

        pre_peak_svm = svm_data[peak_index-200: peak_index]
        pre_peak_acc_x = acc_x_data[peak_index-200: peak_index]
        pre_peak_acc_y = acc_y_data[peak_index-200: peak_index]
        pre_peak_acc_z = acc_z_data[peak_index-200: peak_index]

        data_pre_peak['fx'] = pd.Series(pre_peak_acc_x).dropna()
        data_pre_peak['fy'] = pd.Series(pre_peak_acc_y).dropna()
        data_pre_peak['fz'] = pd.Series(pre_peak_acc_z).dropna()
        data_pre_peak['svm'] = pd.Series(pre_peak_svm).dropna()

        data_pre_peak.to_csv(self.new_path + '/' + sample_id + '/' + 'pre_peak.csv')

        post_peak_svm = svm_data[peak_index: peak_index+201]
        post_peak_acc_x = acc_x_data[peak_index: peak_index+201]
        post_peak_acc_y = acc_y_data[peak_index: peak_index+201]
        post_peak_acc_z = acc_z_data[peak_index: peak_index+201]

        data_post_peak['fx'] = pd.Series(post_peak_acc_x).dropna()
        data_post_peak['fy'] = pd.Series(post_peak_acc_y).dropna()
        data_post_peak['fz'] = pd.Series(post_peak_acc_z).dropna()
        data_post_peak['svm'] = pd.Series(post_peak_svm).dropna()

        data_post_peak.to_csv(self.new_path + '/' + sample_id + '/' + 'post_peak.csv')

        pre_impact_svm = svm_data[0: peak_index-200]
        pre_impact_acc_x = acc_x_data[0: peak_index-200]
        pre_impact_acc_y = acc_y_data[0: peak_index-200]
        pre_impact_acc_z = acc_z_data[0: peak_index-200]

        data_pre_impact['fx'] = pd.Series(pre_impact_acc_x).dropna()
        data_pre_impact['fy'] = pd.Series(pre_impact_acc_y).dropna()
        data_pre_impact['fz'] = pd.Series(pre_impact_acc_z).dropna()
        data_pre_impact['svm'] = pd.Series(pre_impact_svm).dropna()

        data_pre_impact.to_csv(self.new_path + '/' + sample_id + '/' + 'pre_impact.csv')

        post_impact_svm = svm_data[peak_index+201:]
        post_impact_acc_x = acc_x_data[peak_index+201:]
        post_impact_acc_y = acc_y_data[peak_index+201:]
        post_impact_acc_z = acc_z_data[peak_index+201:]

        data_post_impact['fx'] = pd.Series(post_impact_acc_x).dropna()
        data_post_impact['fy'] = pd.Series(post_impact_acc_y).dropna()
        data_post_impact['fz'] = pd.Series(post_impact_acc_z).dropna()
        data_post_impact['svm'] = pd.Series(post_impact_svm).dropna()

        data_post_impact.to_csv(self.new_path + '/' + sample_id + '/' + 'post_impact.csv')

        return {}, True

'''
# Start
features = []
fall = False

data = DataManager('F04_SA01_R01.txt', 1, True)
feature, found = data.get_sis_fall_params()
'''
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
                for type in ['F']:
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
                        data = DataManager(filename, person, fall)
                        feature, found = data.get_sis_fall_params()
                    except ValueError:
                        pass

