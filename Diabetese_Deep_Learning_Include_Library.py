# -------------------------------------------------------------------------------
# Name:        Diabetese_Deep_Learning_Include_Library
# Purpose:     Diabetese diagnosisi with deep learning
# Author:      Saberi
# Created:     20 February 2023
# Licence:     licenced by Saberi
# -------------------------------------------------------------------------------
import math
import seaborn
import numpy as np
import pandas as pd
from numpy import loadtxt
from scipy import interpolate
from tkinter import messagebox
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.impute import SimpleImputer
from sklearn.datasets import load_digits
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import Normalizer
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
Diabetese_CSV_Dataset = 'Diabetese_Dataset.csv'
Diabetese_Negative_Worksheet = 'Diabetese_Negative'
Diabetese_Positive_Worksheet = 'Diabetese_Positive'
Diabetese_Negative_Dataset = 'Diabetese_Negative.xlsx'
Diabetese_Positive_Dataset = 'Diabetese_Positive.xlsx'
Interval = 2000  # in mili second
