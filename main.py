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
plt.savefig('BBC-distribution.pdf', dpi=320)
plt.show()

# 3 load files with encoding latin1
BBC_data_raw = load_files("inputData/BBC/", load_content=True, encoding="latin1")
# TEST
# print(BBC_data_raw.target_names)
# print(BBC_data_raw.data)
# print(BBC_data_raw.target[2003])
# print(BBC_data_raw.data[2003])
# print(BBC_data_raw.target_names[2])
# END_TEST

# 4 pre-process dataset to have the features ready to be used for NB



# part1 examples of functions
# data = load_files()
# fig.savefig('comparison.png', dpi=200)

# part2 examples of functions
# df = pd.read_csv('data.csv')
# pd.get_dummies(df, columns=['name'])
# pd.Categorical([1, 2, 3, 1, 2, 3])
