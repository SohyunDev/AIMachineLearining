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

def column(matrix, i):
    return [row[i] for row in matrix]

data, answer = readFile()

#Handle NaN data with imputer
imp = Imputer(missing_values='NaN', strategy='mean')
imputed_data = imp.fit_transform(data)

standard_imputed_data = []
for index in range(0,374):
    col = column(imputed_data, index)
    standard_imputed_data.append(np.std(col))
print(standard_imputed_data)

train_data, test_data, train_answer, test_answer = crossValidation(imputed_data, answer)

nbrs= KNeighborsClassifier(n_neighbors=5)

PCA_accuracy_train = []
PCA_accuracy_test = []
PCA_accuracy_average = []
max_accuracy_index = 1
maximum = 0

#1/1000~1까지 없앴을때
for index in range(1,500):
    del_index = []
    bound = float(index)/1000;
    for standard_index in range(0,len(standard_imputed_data)):
        if standard_imputed_data[standard_index]<bound:
            del_index.append(standard_index)
    del_train_data = np.delete(train_data, del_index, 1)
    del_test_data = np.delete(test_data, del_index, 1)

    # Preprocessing (Standardzation)
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler().fit(del_train_data.astype(float))
    standard_train_data = scaler.transform(del_train_data.astype(float))
    scaler = preprocessing.StandardScaler().fit(del_test_data.astype(float))
    standard_test_data = scaler.transform(del_test_data.astype(float))

    #Feature Selection PCA
    from sklearn.decomposition import PCA

    pca = PCA(n_components=10)
    pca.fit(standard_train_data)
    train_data_proc = pca.transform(standard_train_data)
    test_data_proc = pca.transform(standard_test_data)
    train_model = nbrs.fit(train_data_proc, train_answer)
    test_pred = train_model.predict(test_data_proc)
    correct_count = (test_pred == test_answer).sum()
    train_accuracy = correct_count / len(test_answer)
    PCA_accuracy_train.append(train_accuracy)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

PCA_accuracy_train2 = []
PCA_accuracy_test2 = []
PCA_accuracy_average2 = []
max_accuracy_index2 = 1
maximum2 = 0

#1/1000~1까지 없앴을때
for index in range(1,500):
    del_index2 = []
    bound = float(index)/1000;
    for standard_index in range(0,len(standard_imputed_data)):
        if standard_imputed_data[standard_index]<bound:
            del_index2.append(standard_index)
    del_train_data = np.delete(train_data, del_index2, 1)
    del_test_data = np.delete(test_data, del_index2, 1)

    # Preprocessing (Standardzation)
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler().fit(del_train_data.astype(float))
    standard_train_data = scaler.transform(del_train_data.astype(float))
    scaler = preprocessing.StandardScaler().fit(del_test_data.astype(float))
    standard_test_data = scaler.transform(del_test_data.astype(float))

    #Feature Selection PCA
    from sklearn.decomposition import PCA

    pca = PCA(n_components=10)
    pca.fit(standard_train_data)
    train_data_proc = pca.transform(standard_train_data)
    test_data_proc = pca.transform(standard_test_data)
    train_model = nb.fit(train_data_proc, train_answer)
    test_pred = train_model.predict(test_data_proc)
    correct_count = (test_pred == test_answer).sum()
    train_accuracy = correct_count / len(test_answer)
    PCA_accuracy_train2.append(train_accuracy)
'''
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

set_PCA_accuracy = set(PCA_accuracy_test)
set_PCA_accuracy=list(set_PCA_accuracy)
number = [0]*len(set_PCA_accuracy)
for index in range(0, len(PCA_accuracy_test)):
    for accuracy_index in range(0,len(set_PCA_accuracy)):
        if(set_PCA_accuracy[accuracy_index] == PCA_accuracy_test[index]):
            number[accuracy_index] += 1

max = 0
max_index =0
for index in range(0,len(number)):
    if max<number[index]:
        max_index = index
        max = number[index]

del_index =[]
for index in range(0,len(PCA_accuracy_test)):
    if max<PCA_accuracy_test[index]:
        del_index.append(index)

del_train_data = np.delete(train_data, [50], 1);
del_test_data = np.delete(test_data, [50], 1);
pca = PCA(n_components=10)
pca.fit(del_train_data)
train_data_proc = pca.transform(del_train_data)
test_data_proc = pca.transform(del_test_data)
train_model = nbrs.fit(train_data_proc, train_answer)
test_pred = train_model.predict(test_data_proc)
correct_count = (test_pred == test_answer).sum()
train_accuracy = correct_count / len(test_answer)
print("del accuracy : "+str(train_accuracy))



print(max_accuracy_index)
print(maximum)
'''
plot_index = range(1,500)
plt.plot(plot_index, PCA_accuracy_train, label="Accuracy_train")
plt.plot(plot_index, PCA_accuracy_train2, label="Accuracy_train")
#plt.plot(feature_size, PCA_accuracy_test, label="Accuracy_test")
#plt.plot(feature_size, PCA_accuracy_average, label="Accuracy_average")
plt.ylabel("Accuracy")
plt.xlabel("std_bound(/1000)")
plt.show()
