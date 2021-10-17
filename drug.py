import warnings

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import math

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from tabulate import tabulate

warnings.filterwarnings('ignore')
# Read the csv file
file_addr = r'inputData\Drug\drug200.csv'
dataframe = pd.read_csv(file_addr)

# Plot the data of drug200.csv

plt.style.use('classic')
drug_group = [x for x in dataframe['Drug']]
drug_type_count = []
drug_type = ['drugA', 'drugB', 'drugC', 'drugX', 'drugY']
for x in drug_type:
    drug_type_count.append(drug_group.count(x))
x = np.array(drug_type)
y = np.array(drug_type_count)
plt.bar(x, y, color="#FFCB6B", width=0.2)
plt.savefig('Drug-distribution.pdf', dpi=320)

# Convert all ordinal and nominal data to data in numeric format
# dataframe['Drug'].replace({'drugA': 1, 'drugB': 2, 'drugC': 3, 'drugX': 4, 'drugY': 5}, inplace=True)
dataframe['Drug'].replace({'drugA': -2, 'drugB': -1, 'drugC': 0, 'drugX': 1, 'drugY': 2}, inplace=True)
dataframe['Sex'].replace({'F': 0, 'M': 1}, inplace=True)
dataframe['BP'].replace({'LOW': 1, 'NORMAL': 2, 'HIGH': 3}, inplace=True)
dataframe['Cholesterol'].replace({'LOW': 1, 'NORMAL': 2, 'HIGH': 3}, inplace=True)

drug_data = dataframe.values[:, :-1]
drug_classes = dataframe.values[:, -1]

# drug_classes_p = pd.get_dummies(dataframe['Drug']).values
# print(drug_classes_p)

# split the data set
drug_train, drug_test, drug_train_target, drug_test_target = train_test_split(drug_data, drug_classes)
# drug_train_p, drug_test_p, drug_train_target_p, drug_test_target_p = train_test_split(drug_data, drug_classes_p)

# Task2_6 Run 6 different classifiers
f = open("drug-performance.txt", 'w')
X = drug_train
y = drug_train_target
test_X = drug_test
class_columns = ['drugA', 'drugB', 'drugC', 'drugX', 'drugY']
feature_columns = ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']
# (a) NB: a Gaussian Naive Bayes Classifier (naive bayes.GaussianNB) with the default parameters.
f.write("(a) ---------------- GaussianNB default values-------------------\n")
clf1 = GaussianNB()
clf1.fit(X, y)
predict_result_1 = clf1.predict(test_X)

matrix_nb = confusion_matrix(drug_test_target, predict_result_1)
f.write("The Confusion Matrix\n")
cm1 = pd.DataFrame(matrix_nb, index=class_columns)
f.write(tabulate(cm1, class_columns, tablefmt="grid", stralign='center'))
f.write('\n')

classification_report_1 = classification_report(drug_test_target, predict_result_1, target_names=class_columns)
f.write("Precision, recall, and F1-measure for each class\n")
f.write(classification_report_1)
f.write('\naccuracy, macro-average F1 and weighted-average F1\n')
row_Index = ["accuracy", "macro F1", "weighted F1"]
accuracy = str(accuracy_score(drug_test_target, predict_result_1))
macro_f1 = str(f1_score(drug_test_target, predict_result_1, average='macro'))
weighted_f1 = str(f1_score(drug_test_target, predict_result_1, average='weighted'))
displayed_data = pd.DataFrame([accuracy, macro_f1, weighted_f1], row_Index)
f.write(tabulate(displayed_data, tablefmt="grid"))

# (b) Base-DT: a Decision Tree (tree.DecisionTreeClassifier) with the default parameters
f.write("\n(b) ---------------- Base-DT default values-------------------\n")
clf2 = DecisionTreeClassifier()
clf2.fit(X, y)
tree.plot_tree(clf2)
predict_result_2 = clf2.predict(test_X)


matrix_nb = confusion_matrix(drug_test_target, predict_result_2)
class_columns = ['drugA', 'drugB', 'drugC', 'drugX', 'drugY']
f.write("The Confusion Matrix\n")
cm1 = pd.DataFrame(matrix_nb, index=class_columns)
f.write(tabulate(cm1, class_columns, tablefmt="grid", stralign='center'))
f.write('\n')

classification_report_2= classification_report(drug_test_target, predict_result_2, target_names=class_columns)
f.write("Precision, recall, and F1-measure for each class\n")
f.write(classification_report_2)
f.write('\naccuracy, macro-average F1 and weighted-average F1\n')
row_Index = ["accuracy", "macro F1", "weighted F1"]
accuracy = str(accuracy_score(drug_test_target, predict_result_2))
macro_f1 = str(f1_score(drug_test_target, predict_result_2, average='macro'))
weighted_f1 = str(f1_score(drug_test_target, predict_result_2, average='weighted'))
displayed_data = pd.DataFrame([accuracy, macro_f1, weighted_f1], row_Index)
f.write(tabulate(displayed_data, tablefmt="grid"))


# (c) Top-DT: a better performing Decision Tree found using (GridSearchCV). The grid search will allow
# you to find the best combination of hyper-parameters, as determined by the evaluation function that
# you have determined in step (3) above. The hyper-parameters that you will experiment with are:
f.write("\n(c) -----------------Top-DT--------------------\n")
# Experiment 1  Entropy + max_depth(3) + min_sample_split(2)
top_DT_param = {'criterion': ('entropy', 'gini'),
                'max_depth': (5, 10), 'min_samples_split': (2, 4, 6)}
clf_3 = GridSearchCV(DecisionTreeClassifier(), top_DT_param)
clf_3.fit(X, y)
# predict with the best found params
predict_result_3 = clf_3.predict(test_X)

# write in the best param combination
f.write("Best parameter combination: \n")
f.write(str(clf_3.best_params_) + '\n')

cm = confusion_matrix(drug_test_target, predict_result_3)
class_columns = ['drugA', 'drugB', 'drugC', 'drugX', 'drugY']
f.write("The Confusion Matrix\n")
cm1 = pd.DataFrame(cm, index=class_columns)
f.write(tabulate(cm1, class_columns, tablefmt="grid", stralign='center'))
f.write('\n')

classification_report_3 = classification_report(drug_test_target, predict_result_3, target_names=class_columns)
f.write("Precision, recall, and F1-measure for each class\n")
f.write(classification_report_3)
f.write('\naccuracy, macro-average F1 and weighted-average F1\n')
row_Index = ["accuracy", "macro F1", "weighted F1"]
accuracy = str(accuracy_score(drug_test_target, predict_result_3))
macro_f1 = str(f1_score(drug_test_target, predict_result_3, average='macro'))
weighted_f1 = str(f1_score(drug_test_target, predict_result_3, average='weighted'))
displayed_data = pd.DataFrame([accuracy, macro_f1, weighted_f1], row_Index)
f.write(tabulate(displayed_data, tablefmt="grid"))

# (d) PER: a Perceptron (linear model.Perceptron), with default parameter values
f.write("\n(d) ---------------- Perceptron default values-------------------\n")
clf4 = Perceptron()
clf4.fit(X, y)
predict_result_4 = clf4.predict(test_X)


matrix_nb = confusion_matrix(drug_test_target, predict_result_4)
class_columns = ['drugA', 'drugB', 'drugC', 'drugX', 'drugY']
f.write("The Confusion Matrix\n")
cm1 = pd.DataFrame(matrix_nb, index=class_columns)
f.write(tabulate(cm1, class_columns, tablefmt="grid", stralign='center'))
f.write('\n')


f.write("Precision, recall, and F1-measure for each class\n")
classification_report_4 = classification_report(drug_test_target, predict_result_4, target_names=class_columns)
f.write(classification_report_4)
f.write('\naccuracy, macro-average F1 and weighted-average F1\n')
row_Index = ["accuracy", "macro F1", "weighted F1"]
accuracy = str(accuracy_score(drug_test_target, predict_result_4))
macro_f1 = str(f1_score(drug_test_target, predict_result_4, average='macro'))
weighted_f1 = str(f1_score(drug_test_target, predict_result_4, average='weighted'))
displayed_data = pd.DataFrame([accuracy, macro_f1, weighted_f1], row_Index)
f.write(tabulate(displayed_data, tablefmt="grid"))


# (e) Base-MLP: a Multi-Layered Perceptron (neural network.MLPClassifier) with 1 hidden layer of
# 100 neurons, sigmoid/logistic as activation function, stochastic gradient descent, and default values
# for the rest of the parameters.
f.write("\n(e) ---------------- Base-MLP default values-------------------\n")
clf5 = MLPClassifier(hidden_layer_sizes=(100), activation='logistic', solver='sgd', max_iter=5000)
clf5.fit(X, y)
predict_result_5 = clf5.predict(test_X)
# predict_result_5 = clf5.predict(X)

matrix_nb = confusion_matrix(drug_test_target, predict_result_5)
# matrix_nb = confusion_matrix(y, predict_result_5)

class_columns = ['drugA', 'drugB', 'drugC', 'drugX', 'drugY']
f.write("The Confusion Matrix\n")
cm5 = pd.DataFrame(matrix_nb, index=class_columns)
f.write(tabulate(cm5, class_columns, tablefmt="grid", stralign='center'))
f.write('\n')


f.write("Precision, recall, and F1-measure for each class\n")
classification_report_5 = classification_report(drug_test_target, predict_result_5, target_names=class_columns)
# classification_report_5 = classification_report(y, predict_result_5, target_names=class_columns)

f.write(classification_report_5)
f.write('\naccuracy, macro-average F1 and weighted-average F1\n')
row_Index = ["accuracy", "macro F1", "weighted F1"]
accuracy = str(accuracy_score(drug_test_target, predict_result_5))
macro_f1 = str(f1_score(drug_test_target, predict_result_5, average='macro'))
weighted_f1 = str(f1_score(drug_test_target, predict_result_5, average='weighted'))
displayed_data = pd.DataFrame([accuracy, macro_f1, weighted_f1], row_Index)
f.write(tabulate(displayed_data, tablefmt="grid"))


# (f) Top-MLP: a better performing Multi-Layered Perceptron found using grid search. For this, you need
# to experiment with the following parameter values:
f.write("\n(f) ---------------- Top-MLP default values-------------------\n")
top_MLP_params = {'activation': ('sigmoid', 'tanh', 'relu', 'identity'),
                  'hidden_layer_sizes':((30,50), (20,20,20)),
                  'solver':('adam', 'sgd')}
clf_6 = GridSearchCV(MLPClassifier(), top_MLP_params)
clf_6.fit(X, y)
predict_result_6 = clf_6.predict(test_X)
f.write("Best parameter combination: \n")
f.write(str(clf_6.best_params_) + '\n')

cm = confusion_matrix(drug_test_target, predict_result_6)
class_columns = ['drugA', 'drugB', 'drugC', 'drugX', 'drugY']
f.write("The Confusion Matrix\n")
cm1 = pd.DataFrame(cm, index=class_columns)
f.write(tabulate(cm1, class_columns, tablefmt="grid", stralign='center'))
f.write('\n')

classification_report_6 = classification_report(drug_test_target, predict_result_6, target_names=class_columns)
f.write("Precision, recall, and F1-measure for each class\n")
f.write(classification_report_6)
f.write('\naccuracy, macro-average F1 and weighted-average F1\n')
row_Index = ["accuracy", "macro F1", "weighted F1"]
accuracy = str(accuracy_score(drug_test_target, predict_result_6))
macro_f1 = str(f1_score(drug_test_target, predict_result_6, average='macro'))
weighted_f1 = str(f1_score(drug_test_target, predict_result_6, average='weighted'))
displayed_data = pd.DataFrame([accuracy, macro_f1, weighted_f1], row_Index)
f.write(tabulate(displayed_data, tablefmt="grid"))


#Close the file
f.close()

