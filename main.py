import matplotlib.pyplot as plt
import math
import numpy as np
import os

# mentioned in part1
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# mentioned in part2
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier


# part1 - 2
# create a list that contains corresponding count
def walkFiles(folder_li):
    f_count_list = []
    for folder in folder_li:
        count = 0
        for root, dirs, files in os.walk("./inputData/BBC/" + folder):
            for file in files:
                count += 1
        f_count_list.append(count)
    return f_count_list


folder_list = ["business", "entertainment", "politics", "sport", "tech"]
files_num_list = walkFiles(folder_list)
x = np.array(folder_list)
y = np.array(files_num_list)

plt.bar(x, y, color="#602D35", width=0.2)
# plt.savefig('BBC-distribution.pdf', dpi=320)
# plt.show()

# 3 load files with encoding latin1
BBC_data_raw = load_files("inputData/BBC/", load_content=True, encoding="latin1")
# TEST
# print(BBC_data_raw.target_names)
# print(BBC_data_raw.data)
# END_TEST

# 4 pre-process dataset to have the features ready to be used for NB
vectorizer = CountVectorizer()
BBC_data = vectorizer.fit_transform(BBC_data_raw.data).toarray()    # X
# TEST
# print(vectorizer.get_feature_names_out())
# END_TEST

# 5 split
BBC_classes = BBC_data_raw.target  # y
BBC_data_train, BBC_data_test, BBC_classes_train, BBC_classes_test = train_test_split(BBC_data, BBC_classes, random_state=0)

print(len(BBC_data_test))
print(len(BBC_data_test))
print(len(BBC_classes_train))
print(len(BBC_classes_train))

# 6 - 1st try
clf_1 = MultinomialNB()
# train
clf_1.fit(BBC_data_train, BBC_classes_train)
# test
try_1_pred = clf_1.predict(BBC_data_test)
# print out two result arrays
# print(try_1_pred)
# print(BBC_classes_test)
# confusion matrix
print(confusion_matrix(BBC_classes_test, try_1_pred))
# using classification_report:
# precision
# recall
# F1-measure

# rng = np.random.RandomState(1)
# X = rng.randint(5, size=(6, 100))
# y = np.array([1, 2, 3, 4, 5, 6])
# clf = MultinomialNB()
# clf.fit(X, y)
# print(clf.n_features_in_)
