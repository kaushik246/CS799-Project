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

from MobiAct import MobiActDataParser
from SisFall import SisFallDataParser

class DataManagerFinal:
    def __init__(self, mobi_act=False, sis_fall=True, mbient=True, features=True):
        self.mobi_act = mobi_act
        self.sis_fall = sis_fall
        self.mbient = mbient
        self.sis_path = '/Users/kaushikkota/ms_cs_uw_madison/cs_799/CS799-Project/SisFall_dataset/'
        self.mobi_path = '/Users/kaushikkota/ms_cs_uw_madison/cs_799/CS799-Project/MobiAct_Dataset_v2.0/AnnotatedData/'
        self.mbient_path = '/home/pramodbs/799/MbientData/'

        if features:
            self.features = True
            self.raw_data = False
        else:
            self.raw_data = True
            self.features = False
        self.data = []

    def fetch_data(self):
        if self.mobi_act:
            parser_obj = MobiActDataParser(dir_path=self.mobi_path, features=self.features)
            self.data += parser_obj.fetch_data()
        if self.sis_fall:
            parser_obj = SisFallDataParser(dir_path=self.sis_path, features=self.features)
            self.data += parser_obj.fetch_data()
        if self.mbient:
            parser_obj = MbientDataParser(dir_path=self.mbient_path, features=self.features)
            self.data += parser_obj.fetch_data()
        return self.data

obj = DataManagerFinal(features=False, mobi_act=True, sis_fall=False, mbient=True)
data = obj.fetch_data()
