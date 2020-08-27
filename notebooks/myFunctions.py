import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import xgboost as xgb

from sklearn.metrics import roc_curve, auc

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, confusion_matrix, make_scorer, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support,f1_score,fbeta_score
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression 
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn import tree

import warnings
warnings.filterwarnings('ignore')

'--------------------------------'

# function to retrieve evaluation metrics from predictors for testing data
def print_metrics_test(labels_test, preds_test,test_fpr,test_tpr,test_or_train):
    print(f'{test_or_train}')
    print('-' * 25)
    print("Precision Score: {}".format(precision_score(labels_test, preds_test)))
    print("Recall Score: {}".format(recall_score(labels_test, preds_test)))
    print("Accuracy Score: {}".format(accuracy_score(labels_test, preds_test)))
    print("F1 Score: {}".format(f1_score(labels_test, preds_test)))
    print('AUC: {}'.format(auc(test_fpr, test_tpr)))
    
def plot_feature_importances(model,X_data):
    n_features = X_data.shape[1]
    plt.figure(figsize=(8,8))
    plt.barh(range(n_features), model.feature_importances_, align='center') 
    plt.yticks(np.arange(n_features), X_data.columns.values) 
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')