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
dataframe = pd.read_csv(r'inputData\Drug\drug200.csv')
classname = cbook.get_sample_data(r'inputData\Drug\drug200.csv',asfileobj=False)

# Plot the data of drug200.csv

# Convert all ordinal and nominal data to data in numeric format
dataframe['Drug'].replace({'drugA': 1, 'drugB': 2, 'drugC': 3, 'drugX': 4, 'drugY': 5}, inplace=True)
dataframe['Sex'].replace({'F': 0, 'M': 1}, inplace=True)
dataframe['BP'].replace({'LOW': 1, 'NORMAL': 2,'HIGH': 3}, inplace=True)
dataframe['Cholesterol'].replace({'LOW': 1, 'NORMAL': 2,'HIGH': 3}, inplace=True)
print(dataframe)

# split the data set
drug_train, drug_test = train_test_split(dataframe)
print(drug_train.shape)
print(drug_test.shape)

# Task2_6 Run 6 different classifier
f = open("drug-performance.txt", 'w')
train_target_column = drug_train.Drug
test_target_column = drug_test.Drug
class_columns = ['drugA', 'drugB', 'drugC', 'drugX', 'drugY']

# (a) NB: a Gaussian Naive Bayes Classifier (naive bayes.GaussianNB) with the default parameters.
f.write("(a) ---------------- GaussianNB default values-------------------\n")
clf1 = GaussianNB()
clf1.fit(drug_train, train_target_column)
predict_result = clf1.predict(drug_test)

matrix_nb = confusion_matrix(test_target_column, predict_result)
f.write("The Confusion Matrix\n")
cm1 = pd.DataFrame(matrix_nb, index=class_columns)
f.write(tabulate(cm1, class_columns, tablefmt="grid", stralign='center'))
f.write('\n')

classification_report = classification_report(test_target_column, predict_result, target_names=class_columns)
f.write("Precision, recall, and F1-measure for each class\n")
f.write(classification_report)
f.write('\naccuracy, macro-average F1 and weighted-average F1\n')
row_Index = ["accuracy", "macro F1", "weighted F1"]
accuracy = str(accuracy_score(test_target_column, predict_result))
macro_f1 = str(f1_score(test_target_column, predict_result, average='macro'))
weighted_f1 = str(f1_score(test_target_column, predict_result, average='weighted'))
displayed_data = pd.DataFrame([accuracy, macro_f1, weighted_f1], row_Index)
f.write(tabulate(displayed_data, tablefmt="grid"))

# (b) Base-DT: a Decision Tree (tree.DecisionTreeClassifier) with the default parameters
f.write("\n(b) ---------------- Base-DT default values-------------------\n")
clf2 = DecisionTreeClassifier()
clf2.fit(drug_train, train_target_column)
tree.plot_tree(clf2)
predict_result_2 = clf2.predict(drug_test)


matrix_nb = confusion_matrix(test_target_column, predict_result)
class_columns = ['drugA', 'drugB', 'drugC', 'drugX', 'drugY']
f.write("The Confusion Matrix\n")
cm1 = pd.DataFrame(matrix_nb, index=class_columns)
f.write(tabulate(cm1, class_columns, tablefmt="grid", stralign='center'))
f.write('\n')

#classification_report = classification_report(test_target_column, predict_result_2, target_names=class_columns)
f.write("Precision, recall, and F1-measure for each class\n")
f.write(classification_report)
f.write('\naccuracy, macro-average F1 and weighted-average F1\n')
row_Index = ["accuracy", "macro F1", "weighted F1"]
accuracy = str(accuracy_score(test_target_column, predict_result))
macro_f1 = str(f1_score(test_target_column, predict_result, average='macro'))
weighted_f1 = str(f1_score(test_target_column, predict_result, average='weighted'))
displayed_data = pd.DataFrame([accuracy, macro_f1, weighted_f1], row_Index)
f.write(tabulate(displayed_data, tablefmt="grid"))


# (c) Top-DT: a better performing Decision Tree found using (GridSearchCV). The grid search will allow
# you to find the best combination of hyper-parameters, as determined by the evaluation function that
# you have determined in step (3) above. The hyper-parameters that you will experiment with are:

# (d) PER: a Perceptron (linear model.Perceptron), with default parameter values

# (e) Base-MLP: a Multi-Layered Perceptron (neural network.MLPClassifier) with 1 hidden layer of
# 100 neurons, sigmoid/logistic as activation function, stochastic gradient descent, and default values
# for the rest of the parameters.

# (f) Top-MLP: a better performing Multi-Layered Perceptron found using grid search. For this, you need
# to experiment with the following parameter values:








plt.style.use('classic')


