import xlrd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from random import shuffle
import numpy as np
from sklearn.svm import SVC

def feature_process(X):
    X= list(map(lambda row: list(map(lambda col: 0 if col == 'NaN' else col, row)), X))
    X= np.array(X)
    index = [8, 18, 24, 25, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 277, 346, 347, 349, 350, 355, 356]
    #1000
    X = np.delete(X, np.s_[5:6], axis=1)
    X = np.delete(X, np.s_[8:9], axis=1)
    X = np.delete(X, np.s_[18:19], axis=1)
    X = np.delete(X, np.s_[24:26], axis=1)
    #1000
    X = np.delete(X, np.s_[32:33], axis=1)
    #1000
    X = np.delete(X, np.s_[56:57], axis=1)
    #1000
    X = np.delete(X, np.s_[74:75], axis=1)
    #1000
    X = np.delete(X, np.s_[92:93], axis=1)
    #1000
    X = np.delete(X, np.s_[167:168], axis=1)
    #1000
    X = np.delete(X, np.s_[169:170], axis=1)
    #1000
    X = np.delete(X, np.s_[171:172], axis=1)
    X = np.delete(X, np.s_[172:184], axis=1)
    #1000
    X = np.delete(X, np.s_[201:210], axis=1)
    #1000
    X = np.delete(X, np.s_[238:239], axis=1)
    X = np.delete(X, np.s_[249:262], axis=1)
    #1000
    X = np.delete(X, np.s_[275:277], axis=1)
    X = np.delete(X, np.s_[277:278], axis=1)
    #1000
    X = np.delete(X, np.s_[278:287], axis=1)
    X = np.delete(X, np.s_[287:288], axis=1)
    #1000
    X = np.delete(X, np.s_[314:315], axis=1)
    #1000
    X = np.delete(X, np.s_[316:417], axis=1)
    #1000
    X = np.delete(X, np.s_[319:320], axis=1)
    #1000
    X = np.delete(X, np.s_[323:324], axis=1)
    #1000
    X = np.delete(X, np.s_[332:333], axis=1)
    #1000
    X = np.delete(X, np.s_[339:340], axis=1)
    #1000
    X = np.delete(X, np.s_[345:346], axis=1)
    X = np.delete(X, np.s_[346:348], axis=1)
    X = np.delete(X, np.s_[349:351], axis=1)
    X = np.delete(X, np.s_[355:357], axis=1)
    #1000
    X = np.delete(X, np.s_[357:358], axis=1)
    #1000
    X = np.delete(X, np.s_[363:364], axis=1)
    #1000
    X = np.delete(X, np.s_[375:476], axis=1)
    X = np.round(X)
    #print(X)

    #pca = PCA(n_components=10)
    #X= pca.fit_transform(X)
    #X= list(map(lambda row: (row-min(row))/(max(row)-min(row)), X))
    return X

wb= xlrd.open_workbook("Music_style_train.xlsx")

ws_feature= wb.sheet_by_index(0)
ws_label= wb.sheet_by_index(1)

X= []
y= []
for i in range(ws_feature.nrows):
    X.append(ws_feature.row_values(i))
    y.append(ws_label.row_values(i))

X= feature_process(X)

zipped_data= list(zip(X, y))
shuffle(zipped_data)
X, y= zip(*zipped_data)

train_len=int(len(X)*0.8)

y=np.array(y).reshape([-1])

knn_classifier= KNeighborsClassifier(n_neighbors=len(set(y)))
nb_classifier= GaussianNB()
svc_classifier= SVC()

knn_classifier.fit(X[0:train_len], y[0:train_len])
knn_predict= knn_classifier.predict(X[train_len:])

nb_classifier.fit(X[0:train_len], y[0:train_len])
nb_predict= nb_classifier.predict(X[train_len:])

svc_classifier.fit(X[0:train_len], y[0:train_len])
svc_predict= svc_classifier.predict(X[train_len:])

knn_precision= (y[train_len:]==knn_predict).tolist().count(True)/len(y[train_len:])
nb_precision= (y[train_len:]==nb_predict).tolist().count(True)/len(y[train_len:])
svc_precision= (y[train_len:]==svc_predict).tolist().count(True)/len(y[train_len:])


print("knn precision: %s" % (knn_precision))
print("nb precision: %s" % (nb_precision))
print("svc precision: %s" % (svc_precision))

pass