from logging.handlers import DatagramHandler
import numpy as np
import matplotlib as plt
import pandas as pd

dataset = pd.read_csv("Churn_Modelling.csv")
x = dataset.iloc[:, 3:13].values #Variables
y = dataset.iloc[:,13].values  #Expected prediction
print(x)
#preprocessing data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncode_x1 = LabelEncoder()
x[:,1] = labelEncode_x1.fit_transform(x[:,1])
labelEncode_x2 = LabelEncoder()
x[:,2] = labelEncode_x2.fit_transform(x[:,2])
oneHotEncoder = OneHotEncoder(categorical_features = [0])
x = oneHotEncoder.fit_transform(x).toarray()