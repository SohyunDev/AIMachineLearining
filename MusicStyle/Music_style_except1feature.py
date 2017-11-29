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

# Preprocessing (Standardzation)
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(imputed_data.astype(float))
standard_data = scaler.transform(imputed_data.astype(float))

train_data, test_data, train_answer, test_answer = crossValidation(standard_data, answer)

nbrs= KNeighborsClassifier(n_neighbors=5)

PCA_accuracy_train = []
PCA_accuracy_test = []
PCA_accuracy_average = []
max_accuracy_index = 1
maximum = 0

#Feature Selection PCA
from sklearn.decomposition import PCA
feature_size = range(0,374)
for feature in feature_size:
    del_train_data = np.delete(train_data, feature, 1);
    del_test_data = np.delete(test_data, feature, 1);
    pca = PCA(n_components=10)
    pca.fit(del_train_data)
    train_data_proc = pca.transform(del_train_data)
    test_data_proc = pca.transform(del_test_data)
    train_model = nbrs.fit(train_data_proc, train_answer)
    test_pred = train_model.predict(test_data_proc)
    correct_count = (test_pred == test_answer).sum()
    train_accuracy = correct_count / len(test_answer)
    PCA_accuracy_train.append(train_accuracy)
    test_model = nbrs.fit(test_data_proc, test_answer)
    train_pred = test_model.predict(train_data_proc)
    correct_count = (train_pred == train_answer).sum()
    test_accuracy = correct_count / len(train_answer)
    PCA_accuracy_test.append(test_accuracy)
    average = (train_accuracy+test_accuracy)/2
    PCA_accuracy_average.append(average)
    if train_accuracy>maximum:
        maximum = train_accuracy
        max_accuracy_index = feature

print(max_accuracy_index)
print(maximum)
plt.plot(feature_size, PCA_accuracy_train, label="Accuracy_train")
plt.plot(feature_size, PCA_accuracy_test, label="Accuracy_test")
plt.plot(feature_size, PCA_accuracy_average, label="Accuracy_average")
plt.ylabel("Accuracy")
plt.xlabel("feature_size")
plt.show()
