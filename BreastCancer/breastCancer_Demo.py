import np as np
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

def readFile(dataFileName, testFileName):
    data_xl = pd.ExcelFile(dataFileName)
    test_xl = pd.ExcelFile(testFileName)
    data_excel = pd.read_excel(io=data_xl, sheetname="train", header=None)
    answer_excel = pd.read_excel(io=data_xl, sheetname="answer", header=None)
    test_data_excel = pd.read_excel(io=test_xl, sheet_name="train", header=None)
    trainData = np.array(data_excel.values)
    trainAnswer = np.array(answer_excel.values).flatten().transpose()
    testData = np.array(test_data_excel.values)
    return trainData, trainAnswer, testData

dataFileName = input("dataFileName : ")
testFileName = input("testFileName : ")

trainData, trainAnswer, testData = readFile(dataFileName,testFileName)

nbrs = KNeighborsClassifier(n_neighbors=5)
'''
pca = PCA(n_components=1)
pca.fit(trainData)
train_data_proc=pca.transform(trainData)
test_data_proc = pca.transform(testData)
train_model = nbrs.fit(train_data_proc, trainAnswer)
results = train_model.predict(test_data_proc)
'''
train_model = nbrs.fit(trainData, trainAnswer)
results = train_model.predict(testData)

f = open("result.txt", 'w')
for result in results:
    print(result)
    f.write(str(result))
    f.write("\n")
f.close()



