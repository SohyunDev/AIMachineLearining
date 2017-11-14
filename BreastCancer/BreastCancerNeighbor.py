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
    answer = np.array(answer_excel.values)
    return data, answer

def crossValidation(data, answer):
    train_data, test_data, train_answer, test_answer = train_test_split(data, answer, test_size=0.2)
    return train_data, test_data, train_answer, test_answer



data, answer = readFile()
train_data, test_data, train_answer, test_answer = crossValidation(data, answer)


training_accuracy = []
test_accuracy =[]
neighbors_settings = range(1, 15)

for n_neighbors in neighbors_settings:
    print(n_neighbors)
    nbrs = KNeighborsClassifier(n_neighbors=n_neighbors)
    nbrs.fit(train_data, train_answer)
    #훈련 정확도 저장
    training_accuracy.append(nbrs.score(train_data, train_answer))
    #일반화 정확도 저장
    test_accuracy.append(nbrs.score(test_data, test_answer))


plt.plot(neighbors_settings, training_accuracy, label="train Accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")

plt.show()