# Momona Yamagami

global_path = "D:\\Research\\OneDrive - UW\\Data\\UW biosignal gesture study\\without disability\\" # home
temp_path = "D:\\Research\\data_temp\\"
# global_path = "C:\\Users\\my13\\OneDrive - UW\\Data\\UW biosignal gesture study\\with disability\\" # UW
#     path = "C:\\Users\\my13\\Documents\\Research\\biosignals_data_testing\\"+participantID+"\\"

muscles=['R_bic','R_tri','R_delt','R_trap','L_bic','L_tri','L_delt','L_trap','R_ext','R_flex','R_FDI','R_APB','L_ext','L_flex','L_FDI','L_APB']
muscles_ordered = ['R_trap','L_trap','R_delt','L_delt','R_bic','L_bic','R_tri','L_tri',
                  'R_ext','L_ext','R_flex','L_flex','R_FDI','L_FDI','R_APB','L_APB']
fmts = ['pdf']
# things to import
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import glob
import copy
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy import signal
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from sklearn.decomposition import PCA
import seaborn as sns
from multi_channel_dollar import MultiDollar
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import wilcoxon,ttest_rel
import numpy.matlib
import seaborn as sns
from scipy.stats import zscore


# plt.style.use('dark_background')
# for experimenter-defined
pIDs = ['P101','P102','P103','P104','P105','P106','P107','P108','P109','P110','P111',
        'P112','P114','P115','P116','P118','P119','P121','P122','P123','P124','P125',
        'P126','P127','P128','P131','P132']

pIDs_unimpaired = ['P001','P003','P004','P005','P006','P008','P010','P011']

pIDs_user_defined = np.asarray(['P102','P103','P104','P105','P106','P107','P108','P109','P110',
                    'P111','P112','P114','P115','P116','P118','P119','P121','P122','P123',
                    'P125','P126','P127','P128','P131','P132'])

usr_def_order = np.asarray([0,1,5,6,11,12,14,16,17,20,21,22,23,3,4,7,8,10,18,9,15,2,13,19,24])

dis_usr_def = np.asarray(['SCI 1','SCI 2','other 1','MD 1','MD 2','SCI 3','SCI 4','MD 3','PN 1','ET 1','PN 2','SCI 5','SCI 6',
              'other 2','SCI 7','ET 2','SCI 8','SCI 9','PN 3','other 3','SCI 10','SCI 11','SCI 12','SCI 13','other 4'])

# P113 did not finish the trials
