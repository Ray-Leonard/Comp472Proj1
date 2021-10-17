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
dataframe['Drug'].replace({'drugA': 1, 'drugB': 2, 'drugC': 3, 'drugX': 4, 'drugY': 5}, inplace=True)
# dataframe['Drug'].replace({'drugA': 10, 'drugB': 26, 'drugC': 33, 'drugX': 24, 'drugY': 15}, inplace=True)

dataframe['Sex'].replace({'F': 0, 'M': 1}, inplace=True)
dataframe['BP'].replace({'LOW': 1, 'NORMAL': 2, 'HIGH': 3}, inplace=True)
dataframe['Cholesterol'].replace({'LOW': 1, 'NORMAL': 2, 'HIGH': 3}, inplace=True)

drug_data = dataframe.values[:, :-1]
drug_classes = dataframe.values[:, -1]

# split the data set
drug_train, drug_test, drug_train_target, drug_test_target = train_test_split(drug_data, drug_classes)

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
f.write("\n(c) -----------------Top-DT {Entropy + Max Depth: [3] + Min Sample split: [2]}--------------------\n")
# Experiment 1  Entropy + max_depth(3) + min_sample_split(2)
tree_para_1 = {'criterion': ['entropy'], 'max_depth': [3], 'min_samples_split': [2]}
clf3_1 = GridSearchCV(DecisionTreeClassifier(), tree_para_1)
clf3_1.fit(X, y)
# tree.plot_tree(clf3)
predict_result_3_1 = clf3_1.predict(test_X)



cm = confusion_matrix(drug_test_target, predict_result_3_1)
class_columns = ['drugA', 'drugB', 'drugC', 'drugX', 'drugY']
f.write("The Confusion Matrix\n")
cm1 = pd.DataFrame(cm, index=class_columns)
f.write(tabulate(cm1, class_columns, tablefmt="grid", stralign='center'))
f.write('\n')

classification_report_3_1= classification_report(drug_test_target, predict_result_3_1, target_names=class_columns)
f.write("Precision, recall, and F1-measure for each class\n")
f.write(classification_report_3_1)
f.write('\naccuracy, macro-average F1 and weighted-average F1\n')
row_Index = ["accuracy", "macro F1", "weighted F1"]
accuracy = str(accuracy_score(drug_test_target, predict_result_3_1))
macro_f1 = str(f1_score(drug_test_target, predict_result_3_1, average='macro'))
weighted_f1 = str(f1_score(drug_test_target, predict_result_3_1, average='weighted'))
displayed_data = pd.DataFrame([accuracy, macro_f1, weighted_f1], row_Index)
f.write(tabulate(displayed_data, tablefmt="grid"))

f.write("\n------------------- Top-DT {Entropy + Max Depth: [4] + Min Sample split: [2]}-------------------\n")
# Experiment 2  Entropy + max_depth(4) + min_sample_split(2)
tree_para_2 = {'criterion': ['entropy'], 'max_depth': [4], 'min_samples_split': [2]}
clf3_2 = GridSearchCV(DecisionTreeClassifier(), tree_para_2)
clf3_2.fit(X, y)
# tree.plot_tree(clf3)
predict_result_3_2 = clf3_2.predict(test_X)


cm = confusion_matrix(drug_test_target, predict_result_3_1)
class_columns = ['drugA', 'drugB', 'drugC', 'drugX', 'drugY']
f.write("The Confusion Matrix\n")
cm1 = pd.DataFrame(cm, index=class_columns)
f.write(tabulate(cm1, class_columns, tablefmt="grid", stralign='center'))
f.write('\n')

classification_report_3_2 = classification_report(drug_test_target, predict_result_3_2, target_names=class_columns)
f.write("Precision, recall, and F1-measure for each class\n")
f.write(classification_report_3_2)
f.write('\naccuracy, macro-average F1 and weighted-average F1\n')
row_Index = ["accuracy", "macro F1", "weighted F1"]
accuracy = str(accuracy_score(drug_test_target, predict_result_3_2))
macro_f1 = str(f1_score(drug_test_target, predict_result_3_2, average='macro'))
weighted_f1 = str(f1_score(drug_test_target, predict_result_3_2, average='weighted'))
displayed_data = pd.DataFrame([accuracy, macro_f1, weighted_f1], row_Index)
f.write(tabulate(displayed_data, tablefmt="grid"))

f.write("\n------------------- Top-DT {Entropy + Max Depth: [4] + Min Sample split: [3]}-------------------\n")
# Experiment 3  Entropy + max_depth(4) + min_sample_split(3)
tree_para_3 = {'criterion': ['entropy'], 'max_depth': [4], 'min_samples_split': [3]}
clf3_3 = GridSearchCV(DecisionTreeClassifier(), tree_para_3)
clf3_3.fit(X, y)
# tree.plot_tree(clf3)
predict_result_3_3 = clf3_3.predict(test_X)


cm = confusion_matrix(drug_test_target, predict_result_3_3)
class_columns = ['drugA', 'drugB', 'drugC', 'drugX', 'drugY']
f.write("The Confusion Matrix\n")
cm1 = pd.DataFrame(cm, index=class_columns)
f.write(tabulate(cm1, class_columns, tablefmt="grid", stralign='center'))
f.write('\n')

classification_report_3_3 = classification_report(drug_test_target, predict_result_3_3, target_names=class_columns)
f.write("Precision, recall, and F1-measure for each class\n")
f.write(classification_report_3_3)
f.write('\naccuracy, macro-average F1 and weighted-average F1\n')
row_Index = ["accuracy", "macro F1", "weighted F1"]
accuracy = str(accuracy_score(drug_test_target, predict_result_3_3))
macro_f1 = str(f1_score(drug_test_target, predict_result_3_3, average='macro'))
weighted_f1 = str(f1_score(drug_test_target, predict_result_3_3, average='weighted'))
displayed_data = pd.DataFrame([accuracy, macro_f1, weighted_f1], row_Index)
f.write(tabulate(displayed_data, tablefmt="grid"))


f.write("\n------------------- Top-DT {Entropy + Max Depth: [4] + Min Sample split: [4]}-------------------\n")
# Experiment 4  Entropy + max_depth(4) + min_sample_split(4)
tree_para_4 = {'criterion': ['entropy'], 'max_depth': [4], 'min_samples_split': [4]}
clf3_4 = GridSearchCV(DecisionTreeClassifier(), tree_para_4)
clf3_4.fit(X, y)
# tree.plot_tree(clf3)
predict_result_3_4 = clf3_4.predict(test_X)


cm = confusion_matrix(drug_test_target, predict_result_3_4)
class_columns = ['drugA', 'drugB', 'drugC', 'drugX', 'drugY']
f.write("The Confusion Matrix\n")
cm1 = pd.DataFrame(cm, index=class_columns)
f.write(tabulate(cm1, class_columns, tablefmt="grid", stralign='center'))
f.write('\n')

classification_report_3_4 = classification_report(drug_test_target, predict_result_3_4, target_names=class_columns)
f.write("Precision, recall, and F1-measure for each class\n")
f.write(classification_report_3_4)
f.write('\naccuracy, macro-average F1 and weighted-average F1\n')
row_Index = ["accuracy", "macro F1", "weighted F1"]
accuracy = str(accuracy_score(drug_test_target, predict_result_3_4))
macro_f1 = str(f1_score(drug_test_target, predict_result_3_4, average='macro'))
weighted_f1 = str(f1_score(drug_test_target, predict_result_3_4, average='weighted'))
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


matrix_nb = confusion_matrix(drug_test_target, predict_result_5)
class_columns = ['drugA', 'drugB', 'drugC', 'drugX', 'drugY']
f.write("The Confusion Matrix\n")
cm5 = pd.DataFrame(matrix_nb, index=class_columns)
f.write(tabulate(cm5, class_columns, tablefmt="grid", stralign='center'))
f.write('\n')


f.write("Precision, recall, and F1-measure for each class\n")
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


#Close the file
f.close()

