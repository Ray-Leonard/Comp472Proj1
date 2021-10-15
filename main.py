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
plt.savefig('BBC-distribution.pdf', dpi=320)


# 3 load files with encoding latin1
BBC_data_raw = load_files("inputData/BBC/", load_content=True, encoding="latin1")


# 4 pre-process dataset to have the features ready to be used for NB
vectorizer = CountVectorizer()
BBC_data = vectorizer.fit_transform(BBC_data_raw.data).toarray()    # X


# 5 split
BBC_classes = BBC_data_raw.target  # y
BBC_data_train, BBC_data_test, BBC_classes_train, BBC_classes_test = train_test_split(BBC_data, BBC_classes, random_state=0, train_size=0.8)


# generating report
f = open("bbc-performance.txt", 'w')
f.write("(a) ---------------- MultinomialNB default values, try 1 ---------------\n")

#################################### 1st try ###########################################################################
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
row_Index_7i=row_Index_7e
for i in range(len(clf_1.feature_count_)):
    n = 0
    for j in range(len(clf_1.feature_count_[i])):
        if clf_1.feature_count_[i][j] == 0:
            n += 1
    number_0.append(n)
    percentage_0.append(n / len(clf_1.feature_count_[i]))

ans7i_percent = pd.DataFrame(percentage_0, row_Index_7i)
ans7i_num = pd.DataFrame(number_0, row_Index_7i)
f.write("[Number]:\n")
f.write(tabulate(ans7i_num, tablefmt='grid'))
f.write("\n\n[Percentage]:\n")
f.write(tabulate(ans7i_percent, tablefmt='grid'))



f.write('\n(j) the number and percentage of words with a frequency of one in the entire corpus\n')
number_1 = 0
for i in sum(clf_1.feature_count_):
    if i == 1:
        number_1 += 1
percentage_1 = number_1 / len(clf_1.feature_count_[0])
f.write("percentage: " + str(percentage_1) + '\n')
f.write("count: " + str(number_1) + '\n')

# two favorite words and their log-prob
vocabulary_list = vectorizer.get_feature_names_out()
favorite_index_1 = np.where(vocabulary_list == 'potato')[0][0]
favorite_index_2 = np.where(vocabulary_list == 'find')[0][0]
f.write("\n(k) 2 favorite words and their log-prob\n")
row_Index_7k = row_Index_7e
f.write("potato: \n")
business = clf_1.feature_log_prob_[0][favorite_index_1]
entertainment = clf_1.feature_log_prob_[1][favorite_index_1]
politics = clf_1.feature_log_prob_[2][favorite_index_1]
sport = clf_1.feature_log_prob_[3][favorite_index_1]
tech = clf_1.feature_log_prob_[4][favorite_index_1]
ans7k_1 = pd.DataFrame([business, entertainment, politics, sport, tech], row_Index_7k)
f.write(tabulate(ans7k_1, tablefmt='grid'))
f.write('\nfind: \n')
business = clf_1.feature_log_prob_[0][favorite_index_2]
entertainment = clf_1.feature_log_prob_[1][favorite_index_2]
politics = clf_1.feature_log_prob_[2][favorite_index_2]
sport = clf_1.feature_log_prob_[3][favorite_index_2]
tech = clf_1.feature_log_prob_[4][favorite_index_2]
ans7k_2 = pd.DataFrame([business, entertainment, politics, sport, tech], row_Index_7k)
f.write(tabulate(ans7k_2, tablefmt='grid'))

######################################### 8- 2nd try Model2 ############################################################
f.write(" \n\n--------------- MultinomialNB default values, try 2 ---------------\n")
clf_2 = MultinomialNB()
# train
clf_2.fit(BBC_data_train, BBC_classes_train)
# test
try_2_pred = clf_2.predict(BBC_data_test)

# confusion matrix
confusion_matrix_2 = confusion_matrix(BBC_classes_test, try_2_pred)
f.write("(a) The Confusion Matrix\n")
cm2 = pd.DataFrame(confusion_matrix_2, index=folder_list)
f.write(tabulate(cm2, folder_list, tablefmt="grid", stralign='center'))
f.write('\n')

# using classification_report, accuracy_score, and f1_score to get precision, recall and F1-measure
classification_report_2 = classification_report(BBC_classes_test, try_2_pred, target_names=BBC_data_raw.target_names)
f.write("(b) Precision, recall, and F1-measure for each class\n")
f.write(classification_report_2)
f.write('\n(c) accuracy, macro-average F1 and weighted-average F1\n')
row_Index_8d = ["accuracy", "macro F1", "weighted F1"]
accuracy = str(accuracy_score(BBC_classes_test, try_2_pred))
macro_f1 = str(f1_score(BBC_classes_test, try_2_pred, average='macro'))
weighted_f1 = str(f1_score(BBC_classes_test, try_2_pred, average='weighted'))
ans8d = pd.DataFrame([accuracy, macro_f1, weighted_f1], row_Index_8d)
f.write(tabulate(ans8d, tablefmt="grid"))

# prior prob of each class
f.write("\n(d) the prior probability of each class\n")
total_2 = sum(clf_2.class_count_)
row_Index_8e = ["business", "entertainment", "politics", "sport", "tech"]
business = (clf_2.class_count_[0] / total_2)
entertainment = (clf_2.class_count_[1] / total_2)
politics = (clf_2.class_count_[2] / total_2)
sport = (clf_2.class_count_[3] / total_2)
tech = (clf_2.class_count_[4] / total_2)
ans8e = pd.DataFrame([business, entertainment, politics, sport, tech], row_Index_8e)
f.write(tabulate(ans8e, tablefmt="grid"))

# vocabulary size
f.write("\n(e) the size of the vocabulary: " + str(len(clf_2.feature_count_[0])))

# number of word-tokens in each class
f.write("\n(f) number of word-tokens in each class\n")
row_Index_8g = row_Index_8e
business = sum(clf_2.feature_count_[0])
entertainment = sum(clf_2.feature_count_[1])
politics = sum(clf_2.feature_count_[2])
sport = sum(clf_2.feature_count_[3])
tech = sum(clf_2.feature_count_[4])
ans8g = pd.DataFrame([business, entertainment, politics, sport, tech], row_Index_8g)
f.write(tabulate(ans8g, tablefmt='grid'))

f.write('\n(g) number of word-tokens in the entire corpus: ' + str(sum(sum(clf_2.feature_count_))) + '\n')

f.write('\n(h) the number and percentage of words with a frequency of zero in each class\n')
percentage_0 = []
number_0 = []
row_Index_8i=row_Index_8e
for i in range(len(clf_2.feature_count_)):
    n = 0
    for j in range(len(clf_2.feature_count_[i])):
        if clf_2.feature_count_[i][j] == 0:
            n += 1
    number_0.append(n)
    percentage_0.append(n / len(clf_2.feature_count_[i]))
ans8i_percent = pd.DataFrame(percentage_0,row_Index_8i)
ans8i_num = pd.DataFrame(number_0,row_Index_8i)
f.write("[Number]:\n")
f.write(tabulate(ans8i_num, tablefmt='grid'))
f.write("\n[Percentage]:\n")
f.write(tabulate(ans8i_percent, tablefmt='grid'))

f.write('\n(i) the number and percentage of words with a frequency of one in the entire corpus\n')
number_1 = 0
for i in sum(clf_2.feature_count_):
    if i == 1:
        number_1 += 1
percentage_1 = number_1 / len(clf_2.feature_count_[0])
f.write("percentage: " + str(percentage_1) + '\n')
f.write("count: " + str(number_1) + '\n')

# two favorite words and their log-prob
favorite_index_1 = np.where(vocabulary_list == 'potato')[0][0]
favorite_index_2 = np.where(vocabulary_list == 'find')[0][0]
f.write("\n(j) 2 favorite words and their log-prob\n")
row_Index_8k = row_Index_8e
f.write("potato: \n")
business = clf_2.feature_log_prob_[0][favorite_index_1]
entertainment = clf_2.feature_log_prob_[1][favorite_index_1]
politics = clf_2.feature_log_prob_[2][favorite_index_1]
sport = clf_2.feature_log_prob_[3][favorite_index_1]
tech = clf_2.feature_log_prob_[4][favorite_index_1]
ans8k_1 = pd.DataFrame([business, entertainment, politics, sport, tech], row_Index_8k)
f.write(tabulate(ans8k_1, tablefmt='grid'))
f.write('\nfind: \n')
business = clf_2.feature_log_prob_[0][favorite_index_2]
entertainment = clf_2.feature_log_prob_[1][favorite_index_2]
politics = clf_2.feature_log_prob_[2][favorite_index_2]
sport = clf_2.feature_log_prob_[3][favorite_index_2]
tech = clf_2.feature_log_prob_[4][favorite_index_2]
ans8k_2 = pd.DataFrame([business, entertainment, politics, sport, tech], row_Index_8k)
f.write(tabulate(ans8k_2, tablefmt='grid'))

######################################################################################################################
f.write("\n---------------- MultinomialNB smoothing values [0.0001] ---------------\n")
clf_3 = MultinomialNB(alpha=0.0001)
# train
clf_3.fit(BBC_data_train, BBC_classes_train)
# test
try_3_pred = clf_3.predict(BBC_data_test)

# confusion matrix
confusion_matrix_3 = confusion_matrix(BBC_classes_test, try_3_pred)
f.write("(a) The Confusion Matrix\n")
cm3 = pd.DataFrame(confusion_matrix_3, index=folder_list)
f.write(tabulate(cm3, folder_list, tablefmt="grid", stralign='center'))
f.write('\n')

# using classification_report, accuracy_score, and f1_score to get precision, recall and F1-measure
classification_report_3 = classification_report(BBC_classes_test, try_3_pred, target_names=BBC_data_raw.target_names)
f.write("(b) Precision, recall, and F1-measure for each class\n")
f.write(classification_report_3)
f.write('\n(c) accuracy, macro-average F1 and weighted-average F1\n')
row_Index_9d = ["accuracy", "macro F1", "weighted F1"]
accuracy = str(accuracy_score(BBC_classes_test, try_3_pred))
macro_f1 = str(f1_score(BBC_classes_test, try_3_pred, average='macro'))
weighted_f1 = str(f1_score(BBC_classes_test, try_3_pred, average='weighted'))
ans9d = pd.DataFrame([accuracy, macro_f1, weighted_f1], row_Index_9d)
f.write(tabulate(ans9d, tablefmt="grid"))

# prior prob of each class
f.write("\n(d) the prior probability of each class\n")
total_3 = sum(clf_3.class_count_)
row_Index_9e = ["business", "entertainment", "politics", "sport", "tech"]
business = (clf_3.class_count_[0] / total_3)
entertainment = (clf_3.class_count_[1] / total_3)
politics = (clf_3.class_count_[2] / total_3)
sport = (clf_3.class_count_[3] / total_3)
tech = (clf_3.class_count_[4] / total_3)
ans9e = pd.DataFrame([business, entertainment, politics, sport, tech], row_Index_9e)
f.write(tabulate(ans9e, tablefmt="grid"))

# vocabulary size
f.write("\n(e) the size of the vocabulary: " + str(len(clf_3.feature_count_[0])))

# number of word-tokens in each class
f.write("\n(f) number of word-tokens in each class\n")
row_Index_9g = row_Index_9e
business = sum(clf_3.feature_count_[0])
entertainment = sum(clf_3.feature_count_[1])
politics = sum(clf_3.feature_count_[2])
sport = sum(clf_3.feature_count_[3])
tech = sum(clf_3.feature_count_[4])
ans9g = pd.DataFrame([business, entertainment, politics, sport, tech], row_Index_9g)
f.write(tabulate(ans9g, tablefmt='grid'))

f.write('\n(g) number of word-tokens in the entire corpus: ' + str(sum(sum(clf_3.feature_count_))) + '\n')

f.write('\n(h) the number and percentage of words with a frequency of zero in each class\n')
percentage_0 = []
number_0 = []
row_Index_9i = ["business", "entertainment", "politics", "sport", "tech"]
for i in range(len(clf_3.feature_count_)):
    n = 0
    for j in range(len(clf_3.feature_count_[i])):
        if clf_3.feature_count_[i][j] == 0:
            n += 1
    number_0.append(n)
    percentage_0.append(n / len(clf_3.feature_count_[i]))
ans9i_percent = pd.DataFrame(percentage_0,row_Index_9i)
ans9i_num = pd.DataFrame(number_0, row_Index_9i)
f.write("[Number]:\n")
f.write(tabulate(ans9i_num,tablefmt='grid'))
f.write("\n[Percentage]:\n")
f.write(tabulate(ans9i_percent, tablefmt='grid'))


f.write('\n(i) the number and percentage of words with a frequency of one in the entire corpus\n')
number_1 = 0
for i in sum(clf_3.feature_count_):
    if i == 1:
        number_1 += 1
percentage_1 = number_1 / len(clf_3.feature_count_[0])
f.write("percentage: " + str(percentage_1) + '\n')
f.write("count: " + str(number_1) + '\n')

# two favorite words and their log-prob
favorite_index_1 = np.where(vocabulary_list == 'potato')[0][0]
favorite_index_2 = np.where(vocabulary_list == 'find')[0][0]
f.write("\n(j) 2 favorite words and their log-prob\n")
row_Index_9k = row_Index_9e
f.write("potato: \n")
business = clf_3.feature_log_prob_[0][favorite_index_1]
entertainment = clf_3.feature_log_prob_[1][favorite_index_1]
politics = clf_3.feature_log_prob_[2][favorite_index_1]
sport = clf_3.feature_log_prob_[3][favorite_index_1]
tech = clf_3.feature_log_prob_[4][favorite_index_1]
ans9k_1 = pd.DataFrame([business, entertainment, politics, sport, tech], row_Index_9k)
f.write(tabulate(ans9k_1, tablefmt='grid'))
f.write('\nfind: \n')
business = clf_3.feature_log_prob_[0][favorite_index_2]
entertainment = clf_3.feature_log_prob_[1][favorite_index_2]
politics = clf_3.feature_log_prob_[2][favorite_index_2]
sport = clf_3.feature_log_prob_[3][favorite_index_2]
tech = clf_3.feature_log_prob_[4][favorite_index_2]
ans9k_2 = pd.DataFrame([business, entertainment, politics, sport, tech], row_Index_9k)
f.write(tabulate(ans9k_2, tablefmt='grid'))


######################################################################################################################
# generating report
f.write("\n---------------- MultinomialNB smoothing values [0.9] ---------------\n")
clf_4 = MultinomialNB(alpha=0.9)
# train
clf_4.fit(BBC_data_train, BBC_classes_train)
# test
try_4_pred = clf_4.predict(BBC_data_test)

# confusion matrix
confusion_matrix_4 = confusion_matrix(BBC_classes_test, try_4_pred)
f.write("(a) The Confusion Matrix\n")
cm4 = pd.DataFrame(confusion_matrix_4, index=folder_list)
f.write(tabulate(cm4, folder_list, tablefmt="grid", stralign='center'))
f.write('\n')

# using classification_report, accuracy_score, and f1_score to get precision, recall and F1-measure
classification_report_4 = classification_report(BBC_classes_test, try_4_pred, target_names=BBC_data_raw.target_names)
f.write("(b) Precision, recall, and F1-measure for each class\n")
f.write(classification_report_4)
f.write('\n(c) accuracy, macro-average F1 and weighted-average F1\n')
row_Index_10d = ["accuracy", "macro F1", "weighted F1"]
accuracy = str(accuracy_score(BBC_classes_test, try_4_pred))
macro_f1 = str(f1_score(BBC_classes_test, try_4_pred, average='macro'))
weighted_f1 = str(f1_score(BBC_classes_test, try_4_pred, average='weighted'))
ans10d = pd.DataFrame([accuracy, macro_f1, weighted_f1], row_Index_10d)
f.write(tabulate(ans10d, tablefmt="grid"))

# prior prob of each class
f.write("\n(d) the prior probability of each class\n")
total_4 = sum(clf_4.class_count_)
row_Index_10e = ["business", "entertainment", "politics", "sport", "tech"]
business = (clf_4.class_count_[0] / total_4)
entertainment = (clf_4.class_count_[1] / total_4)
politics = (clf_4.class_count_[2] / total_4)
sport = (clf_4.class_count_[3] / total_4)
tech = (clf_4.class_count_[4] / total_4)
ans10e = pd.DataFrame([business, entertainment, politics, sport, tech], row_Index_10e)
f.write(tabulate(ans10e, tablefmt="grid"))

# vocabulary size
f.write("\n(e) the size of the vocabulary: " + str(len(clf_4.feature_count_[0])))

# number of word-tokens in each class
f.write("\n(f) number of word-tokens in each class\n")
row_Index_10g = row_Index_10e
business = sum(clf_4.feature_count_[0])
entertainment = sum(clf_4.feature_count_[1])
politics = sum(clf_4.feature_count_[2])
sport = sum(clf_4.feature_count_[3])
tech = sum(clf_4.feature_count_[4])
ans10g = pd.DataFrame([business, entertainment, politics, sport, tech], row_Index_9g)
f.write(tabulate(ans10g, tablefmt='grid'))

f.write('\n(g) number of word-tokens in the entire corpus: ' + str(sum(sum(clf_4.feature_count_))) + '\n')

f.write('\n(h) the number and percentage of words with a frequency of zero in each class\n')
percentage_0 = []
number_0 = []
row_Index_10i=["business", "entertainment", "politics", "sport", "tech"]
for i in range(len(clf_4.feature_count_)):
    n = 0
    for j in range(len(clf_4.feature_count_[i])):
        if clf_4.feature_count_[i][j] == 0:
            n += 1
    number_0.append(n)
    percentage_0.append(n / len(clf_4.feature_count_[i]))
ans10i_percent = pd.DataFrame(percentage_0,row_Index_10i)
ans10i_num = pd.DataFrame(number_0,row_Index_10i)
f.write("[Number]:\n")
f.write(tabulate(ans10i_num,tablefmt='grid'))
f.write("\n[Percentage]:\n")
f.write(tabulate(ans10i_percent, tablefmt='grid'))


f.write('\n(i) the number and percentage of words with a frequency of one in the entire corpus\n')
number_1 = 0
for i in sum(clf_4.feature_count_):
    if i == 1:
        number_1 += 1
percentage_1 = number_1 / len(clf_4.feature_count_[0])
f.write("percentage: " + str(percentage_1) + '\n')
f.write("count: " + str(number_1) + '\n')

# two favorite words and their log-prob
favorite_index_1 = np.where(vocabulary_list == 'potato')[0][0]
favorite_index_2 = np.where(vocabulary_list == 'find')[0][0]
f.write("\n(j) 2 favorite words and their log-prob\n")
row_Index_10k = row_Index_10e
f.write("potato: \n")
business = clf_4.feature_log_prob_[0][favorite_index_1]
entertainment = clf_4.feature_log_prob_[1][favorite_index_1]
politics = clf_4.feature_log_prob_[2][favorite_index_1]
sport = clf_4.feature_log_prob_[3][favorite_index_1]
tech = clf_4.feature_log_prob_[4][favorite_index_1]
ans10k_1 = pd.DataFrame([business, entertainment, politics, sport, tech], row_Index_10k)
f.write(tabulate(ans10k_1, tablefmt='grid'))
f.write('\nfind: \n')
business = clf_4.feature_log_prob_[0][favorite_index_2]
entertainment = clf_4.feature_log_prob_[1][favorite_index_2]
politics = clf_4.feature_log_prob_[2][favorite_index_2]
sport = clf_4.feature_log_prob_[3][favorite_index_2]
tech = clf_4.feature_log_prob_[4][favorite_index_2]
ans10k_2 = pd.DataFrame([business, entertainment, politics, sport, tech], row_Index_10k)
f.write(tabulate(ans10k_2, tablefmt='grid'))
# remember to close the file!
f.close()