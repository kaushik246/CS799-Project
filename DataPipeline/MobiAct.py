import os
import csv
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
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std

class MobiActDataParser:

    def __init__(self, dir_path, features):
        self.dir_path = dir_path
        self.features = features
        self.participants = 67
        self.trials = 3
        self.falls = ['FOL', 'FKL', 'BSC', 'SDL']
        self.adls = ['CHU', 'CSI', 'CSO', 'JOG', 'JUM', 'SBE', 'SBW', 'SCH', 'SIT', 'SLH', 'SLW', 'SRH', 'STD', 'STN', 'STU',
            'WAL']
        self.data = []

    def fetch_data(self):
        for participant in range(1, self.participants+1):
            for trial in range(1, self.trials+1):
                for fall in self.falls:
                    label = True
                    filename = fall + '_' + str(participant) + '_' + str(trial) + '_annotated.csv'
                    parserObj = MobiActDataManager(filename, self.dir_path + fall + '/', label, self.features)

                    if self.features:
                        feature, found = parserObj.read_data()
                        if found:
                            self.data.append(feature)
                    else:
                        data, found = parserObj.read_data()
                        if found:
                            self.data.append([data, label])

                for adl in self.adls:
                    label = False
                    filename = adl + '_' + str(participant) + '_' + str(trial) + '_annotated.csv'
                    parserObj = MobiActDataManager(filename, self.dir_path + adl + '/', label, self.features)
                    if self.features:
                        feature, found = parserObj.read_data()
                        if found:
                            self.data.append(feature)
                    else:
                        data, found = parserObj.read_data()
                        if found:
                            self.data.append([data, label])
        return self.data


class MobiActDataManager:
    def __init__(self, filename, filepath, label, features):
        self.filename = filename
        self.filepath = filepath
        self.label = label
        self.features = features

    def read_data(self):
        try:
            csv_data = pd.read_csv(self.filepath+self.filename)
            csv_data.drop('timestamp', inplace=True, axis=1)
            csv_data.drop('rel_time', inplace=True, axis=1)
            csv_data.drop('azimuth', inplace=True, axis=1)
            csv_data.drop('pitch', inplace=True, axis=1)
            csv_data.drop('roll', inplace=True, axis=1)
            csv_data.drop('label', inplace=True, axis=1)

            acc_x_data, acc_y_data, acc_z_data = csv_data['acc_x'].to_numpy(), csv_data['acc_y'].to_numpy(), csv_data['acc_z'].to_numpy()
            gyr_x_data, gyr_y_data, gyr_z_data = csv_data['gyro_x'].to_numpy(), csv_data['gyro_y'].to_numpy(), csv_data['gyro_z'].to_numpy()
            acc_data, gyr_data, acc2_data = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

            if not self.features:
                main_data = pd.DataFrame()
                main_data['acc_x'] = pd.Series(acc_x_data).dropna()
                main_data['acc_y'] = pd.Series(acc_y_data).dropna()
                main_data['acc_z'] = pd.Series(acc_z_data).dropna()

                main_data['rot_x'] = pd.Series(gyr_x_data).dropna()
                main_data['rot_y'] = pd.Series(gyr_y_data).dropna()
                main_data['rot_z'] = pd.Series(gyr_z_data).dropna()
                print(main_data)
                return main_data, True

            acc_data['fx'] = pd.Series(self.butterworth_low_pass(acc_x_data, 5.0, 200.0, 4)).dropna()
            acc_data['fy'] = pd.Series(self.butterworth_low_pass(acc_y_data, 5.0, 200.0, 4)).dropna()
            acc_data['fz'] = pd.Series(self.butterworth_low_pass(acc_z_data, 5.0, 200.0, 4)).dropna()

            gyr_data['fx'] = pd.Series(gyr_x_data).dropna()
            gyr_data['fy'] = pd.Series(gyr_y_data).dropna()
            gyr_data['fz'] = pd.Series(gyr_z_data).dropna()

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

            # Gyro Data
            feature_dict['max_rot_x'] = np.max(gyr_data['fx'])
            feature_dict['min_rot_x'] = np.min(gyr_data['fx'])
            feature_dict['mean_rot_x'] = np.mean(gyr_data['fx'])
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

            feature_dict['result'] = 1 if self.label else 0
            return feature_dict, True

            return {}, True
        except FileNotFoundError:
            return {}, False

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

    def butterworth_low_pass(self, data, cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = lfilter(b, a, data)
        return y