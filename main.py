import matplotlib.pyplot as plt
import math
import numpy as np

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


print("Oi boi")

# part1 examples of functions
# data = load_files()

# part2 examples of functions
# df = pd.read_csv('data.csv')
# pd.get_dummies(df, columns=['name'])
# pd.Categorical([1, 2, 3, 1, 2, 3])

