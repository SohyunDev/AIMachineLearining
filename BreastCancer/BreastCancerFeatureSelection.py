import np as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def readFile():
    xl = pd.ExcelFile("Breastcancer_train.xlsx")
    data_excel = pd.read_excel(io=xl, sheetname="train", header=None)
    answer_excel = pd.read_excel(io=xl, sheetname="answer", header=None)
    data = np.array(data_excel.values)
    answer = np.array(answer_excel.values).flatten().transpose()
    return data, answer

def crossValidation(data, answer):
    train_data, test_data, train_answer, test_answer = train_test_split(data, answer, test_size=0.2)
    return train_data, test_data, train_answer, test_answer

def getDataMatrix(data, bit):
    matrix = [[] for _ in range (0, len(data))]

    for index in range(0,len(bit)):
        if bit[index] == '1':
            for idx in range(0,len(data)):
                matrix[idx].append(data[idx][index])
    return matrix

def getAccuracy(data, answer, test_data, test_answer):
    input_list=[]
    accuracy_list = []
    nbrs = KNeighborsClassifier(n_neighbors=5)

    for feature in range(1,2**len(data[0])):
        feature_bit = bin(feature)[2:]
        feature_bit = feature_bit.zfill(9)
        temp_traindata = getDataMatrix(data, feature_bit)
        temp_testdata = getDataMatrix(test_data, feature_bit)
        train_model = nbrs.fit(temp_traindata, answer)
        test_pred = train_model.predict(temp_testdata)
        correct_count = (test_pred == test_answer).sum()
        accuracy = correct_count / len(test_answer)
        input_list.append(feature)
        accuracy_list.append(accuracy)
    return input_list, accuracy_list

data, answer = readFile()
train_data, test_data, train_answer, test_answer = crossValidation(data, answer)

feature_settings = range(0,9)

all_feature_settings = range(1,2**len(data[0]))

input1, train_accuracy_list = getAccuracy(train_data, train_answer, test_data, test_answer)
input2, test_accuracy_list = getAccuracy(test_data,test_answer, train_data, train_answer)

label = range(1,2**len(data[0]))
plt.plot(input1, train_accuracy_list, label="Train_Accuracy")
plt.plot(input2, test_accuracy_list, label="Test_Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("input")
plt.legend()
plt.show()
