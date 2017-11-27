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

train_data, test_data, train_answer, test_answer = crossValidation(imputed_data, answer)

test_accuracy =[]
train_accuracy =[]
aver_accuracy = []
neighbors_settings = range(1, 40)

for n_neighbors in neighbors_settings:
    nbrs= KNeighborsClassifier(n_neighbors=n_neighbors)
    nbrs.fit(train_data,train_answer)
    test_pred = nbrs.predict(test_data)
    correct_count = (test_pred==test_answer).sum()
    test_acc = correct_count/len(test_answer)
    test_accuracy.append(test_acc)

    nbrs.fit(test_data, test_answer)
    train_pred = nbrs.predict(train_data)
    correct_count = (train_pred==train_answer).sum()
    train_acc=correct_count/len(train_answer)
    train_accuracy.append(train_acc)
    aver_accuracy.append((train_acc+test_acc)/2)

plt.plot(neighbors_settings,train_accuracy,label="train_Accuracy")
plt.plot(neighbors_settings,test_accuracy,label="test_Accuracy")
plt.plot(neighbors_settings,aver_accuracy, label="average_Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.show()