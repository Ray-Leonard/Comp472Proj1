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
from tabulate import tabulate

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
BBC_data_train, BBC_data_test, BBC_classes_train, BBC_classes_test = train_test_split(BBC_data, BBC_classes, random_state=0, train_size=0.8)


# generating report
f = open("bbc-performance.txt", 'w')
f.write("(a) ---------------- MultinomialNB default values, try 1 ---------------\n")

# 6 - 1st try
clf_1 = MultinomialNB()
# train
clf_1.fit(BBC_data_train, BBC_classes_train)
# test
try_1_pred = clf_1.predict(BBC_data_test)

# confusion matrix
confusion_matrix_1 = confusion_matrix(BBC_classes_test, try_1_pred)
f.write("(b) The Confusion Matrix\n")
cm1 = pd.DataFrame(confusion_matrix_1, index=folder_list)
f.write(tabulate(cm1, folder_list, tablefmt="grid", stralign='center'))
f.write('\n')

# using classification_report, accuracy_score, and f1_score to get precision, recall and F1-measure
classification_report_1 = classification_report(BBC_classes_test, try_1_pred, target_names=BBC_data_raw.target_names)
f.write("(c) Precision, recall, and F1-measure for each class\n")
f.write(classification_report_1)
f.write('\n(d) accuracy, macro-average F1 and weighted-average F1\n')
row_Index_7d = ["accuracy", "macro F1", "weighted F1"]
accuracy = str(accuracy_score(BBC_classes_test, try_1_pred))
macro_f1 = str(f1_score(BBC_classes_test, try_1_pred, average='macro'))
weighted_f1 = str(f1_score(BBC_classes_test, try_1_pred, average='weighted'))
ans7d = pd.DataFrame([accuracy, macro_f1, weighted_f1], row_Index_7d)
f.write(tabulate(ans7d, tablefmt="grid"))

# prior prob of each class
f.write("\n(e) the prior probability of each class\n")
total_1 = sum(clf_1.class_count_)
row_Index_7e = ["business", "entertainment", "politics", "sport", "tech"]
business = (clf_1.class_count_[0] / total_1)
entertainment = (clf_1.class_count_[1] / total_1)
politics = (clf_1.class_count_[2] / total_1)
sport = (clf_1.class_count_[3] / total_1)
tech = (clf_1.class_count_[4] / total_1)
ans7e = pd.DataFrame([business, entertainment, politics, sport, tech], row_Index_7e)
f.write(tabulate(ans7e, tablefmt="grid"))

# vocabulary size
f.write("\n(f) the size of the vocabulary: " + str(len(clf_1.feature_count_[0])))

# number of word-tokens in each class
f.write("\n(g) number of word-tokens in each class\n")
row_Index_7g = row_Index_7e
business = sum(clf_1.feature_count_[0])
entertainment = sum(clf_1.feature_count_[1])
politics = sum(clf_1.feature_count_[2])
sport = sum(clf_1.feature_count_[3])
tech = sum(clf_1.feature_count_[4])
ans7g = pd.DataFrame([business, entertainment, politics, sport, tech], row_Index_7g)
f.write(tabulate(ans7g, tablefmt='grid'))

f.write('\n(h) number of word-tokens in the entire corpus: ' + str(sum(sum(clf_1.feature_count_))) + '\n')

f.write('\n(i) the number and percentage of words with a frequency of zero in each class\n')
percentage_0 = []
number_0 = []
for i in range(len(clf_1.feature_count_)):
    n = 0
    for j in range(len(clf_1.feature_count_[i])):
        if clf_1.feature_count_[i][j] == 0:
            n += 1
    number_0.append(n)
    percentage_0.append(n / len(clf_1.feature_count_[i]))
# please help me do the formatting and write to file
print(percentage_0)
print(number_0)

f.write('\n(j) the number and percentage of words with a frequency of one in the entire corpus\n')
number_1 = 0
for i in range(len(clf_1.feature_count_[0])):
    count = 0
    for j in range(len(clf_1.feature_count_)):
        count += clf_1.feature_count_[j][i]
    if count == 1:
        number_1 += 1
percentage_1 = number_1 / len(clf_1.feature_count_[0])
print(percentage_1)
print(number_1)


# remember to close the file!
f.close()