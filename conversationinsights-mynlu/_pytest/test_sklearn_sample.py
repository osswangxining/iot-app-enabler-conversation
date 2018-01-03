import numpy as np
import urllib

from sklearn import preprocessing

# url with dataset
#url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
# download the file
#raw_data = urllib.urlopen(url)
# load the CSV file as a numpy matrix
#dataset = np.loadtxt(raw_data, delimiter=",")
dataset = np.loadtxt("/Users/xiningwang/Downloads/pima-indians-diabetes.data.txt", delimiter=",")
# separate the data from the target attributes
X = dataset[:,0:7]
y = dataset[:,8]

# normalize the data attributes
normalized_X = preprocessing.normalize(X)
# standardize the data attributes
standardized_X = preprocessing.scale(X)


from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X, y)
# display the relative importance of each attribute
print("########1.")
print(model.feature_importances_)

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
# create the RFE model and select 3 attributes
rfe = RFE(model, 3)
rfe = rfe.fit(X, y)
# summarize the selection of the attributes
print("########2.")
print(rfe.support_)
print("########3.")
print(rfe.ranking_)

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X, y)
print("########4.")
print(model)
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print("########5.")
print(metrics.classification_report(expected, predicted))
print("########6.")
print(metrics.confusion_matrix(expected, predicted))

from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X, y)
print("########7.")
print(model)
# make predictions
expected = y
predicted = model.predict(X)
# summarize the fit of the model
print("########8.")
print(metrics.classification_report(expected, predicted))
print("########9.")
print(metrics.confusion_matrix(expected, predicted))

sklearn_config = {
        "C": [1, 2, 5, 10, 20, 100],
        "kernel": "linear"
    }
C = sklearn_config.get("C", [1, 2, 5, 10, 20, 100])
kernel = sklearn_config.get("kernel", "linear")
# dirty str fix because sklearn is expecting str not instance of basestr...
tuned_parameters = [{"C": C, "kernel": [str(kernel)]}]
cv_splits = 2
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
classifier = GridSearchCV(SVC(C=1, probability=True, class_weight='balanced'),
                               param_grid=tuned_parameters, n_jobs=1,
                               cv=cv_splits, scoring='f1_weighted', verbose=1)

##classifier.fit(X, y)
print "########10."
###print classifier