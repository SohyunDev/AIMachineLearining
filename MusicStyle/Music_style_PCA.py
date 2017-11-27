import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier


def readFile():
    xl = pd.ExcelFile("Music_style_train.xlsx")
    data_excel = pd.read_excel(io=xl, sheet_name="train_data", header=None)
    answer_excel = pd.read_excel(io=xl, sheet_name="train_label", header=None)
    data= np.array(data_excel.values)
    answer = np.array(answer_excel.values).flatten().transpose()
    return data, answer

def crossValidation(data, answer):
    train_data, test_data, train_answer, test_answer = train_test_split(data, answer, test_size=0.8)
    return train_data, test_data, train_answer, test_answer


data, answer = readFile()

#Handle NaN data with imputer
imp = Imputer(missing_values='NaN', strategy='mean')
imputed_data = imp.fit_transform(data)

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(imputed_data.astype(float))
standard_data = scaler.transform(imputed_data.astype(float))

train_data, test_data, train_answer, test_answer = crossValidation(standard_data, answer)

nbrs= KNeighborsClassifier(n_neighbors=5)
nbrs.fit(train_data,train_answer)
test_pred = nbrs.predict(test_data)
correct_count = (test_pred==test_answer).sum()
test_acc = correct_count/len(test_answer)
print(test_acc)
