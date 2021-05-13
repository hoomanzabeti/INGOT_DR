#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 14:56:07 2020

@author: hoomanzabeti
"""

import cplex
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score
from multiprocessing import Pool
from functools import partial
import copy
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import sys
import itertools
import datetime
import os
import time

from RuleBasedClassifier import drugClassifier
from DataLoader import data_loader

input_drugs = sys.argv[1]
seed = 33
fold_number = 5
test_size = 0.2
cv_refit = 'balanced_accuracy'
drugs = [input_drugs]
FP_list = [0.1]
rule_num_list = [20]
rc_list = [1]
rc_e_list = [1]
rc_p_list = [1e-2,1e-1,1,1e1, 1e2]
rc_z_list = [1e-2,1e-1,1,1e1, 1e2]
cplex_time_limit = [1800]
param_grid = dict(rule_max_length=rule_num_list, FP_up=FP_list, rc=rc_list, rc_e=rc_e_list, rc_p=rc_p_list,
                  rc_z=rc_z_list, cpx_timelimit=cplex_time_limit)

d = datetime.datetime.now()
dir_name = d.strftime("%d") + '_' + d.strftime("%b") + '_' + d.strftime("%Y")
if not os.path.isdir("./Results"):
    path = os.getcwd()
    try:
        os.mkdir(path + "/Results")
    except OSError:
        print("Creation of the directory %s failed" % path + "/Results")
    else:
        print("Successfully created the directory %s " % path + "/Results")
if not os.path.isdir("./Results/" + dir_name):
    path = os.getcwd()
    try:
        os.mkdir(path + "/Results/" + dir_name)
    except OSError:
        print("Creation of the directory %s failed" % path + "/Results/" + dir_name)
    else:
        print("Successfully created the directory%s " % path + "/Results/" + dir_name)


# Loading  the data for drug 'current_drug'


def snps(rule, snp_list):
    rule_to_snps = [snp_list[int(i)] for i in rule]
    return rule_to_snps


def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]


def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]


def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]


def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]


def TNR(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0] / (confusion_matrix(y_true, y_pred)[0, 0] +
                                                                          confusion_matrix(y_true, y_pred)[0, 1])

for current_drug in drugs:
    print('----------------------Data Loading -------------------------------')
    X = pd.read_csv('data/SNPsMatrix_{}.csv'.format(current_drug), index_col=0)
    y = pd.read_csv('data/{}Label.csv'.format(current_drug), index_col=0)[current_drug].tolist()
    snp_list = list(X.columns)
    print('----------------------Train-Test split-------------------------------')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=test_size, stratify=y)

    scoring = dict(Accuracy='accuracy',tp=make_scorer(tp), tn=make_scorer(tn),
                   fp=make_scorer(fp), fn=make_scorer(fn), balanced_accuracy=make_scorer(balanced_accuracy_score))
    start_time = time.time()
    print('----------------------Hyper-parameter Tuning-------------------------------')
    dClassifier = drugClassifier()
    grid = GridSearchCV(dClassifier, param_grid, cv=fold_number, refit=cv_refit, scoring=scoring, n_jobs=-1,
                        return_train_score=True,verbose=10)
    print('----------------------Training-------------------------------')
    grid.fit(X_train, y_train)
    end_time = time.time()
    print('----------------------Saving Cross-Validation Result-------------------------------')
    pd.DataFrame.from_dict(grid.cv_results_).to_csv(r'Results/%s/%s_RB_GridSearchCV.csv' % (dir_name, current_drug))
    pd.DataFrame(grid.best_params_, index=[0]).to_csv(r'Results/%s/%s_RB_BestParameter.csv' % (dir_name, current_drug))
    print('----------------------Validation-------------------------------')
    y_pred = grid.predict(X_test)
    rule = grid.best_estimator_.w_solution_

    report_eval = ['Drug', 'time', 'balanced_accuracy', 'tp', 'tn', 'fp', 'fn', 'sensitivity', 'specificity',
                   'FP_percent', 'Max_rule_size', 'Rule_size', 'rule', 'SNP']

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    test_results = pd.DataFrame(columns=report_eval)
    test_results = test_results.append({'Drug': current_drug, 'time': round(end_time - start_time, 2),
                                        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
                                        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,'sensitivity': tp / (tp + fn),
                                        'specificity': tn / (tn + fp), 'FP_percent': grid.best_estimator_.FP_up_percent,
                                        'Max_rule_size': grid.best_estimator_.rule_max_length, 'Rule_size': len(rule),
                                        'rule': rule, 'SNP': snps(rule=rule, snp_list=snp_list)}, ignore_index=True)
    print('----------{}: Balanced Accuracy{}-----------'.format(current_drug, balanced_accuracy_score(y_test, y_pred)))
    print('----------------------Saving Result-------------------------------')
    test_results.to_csv(r'Results/%s/%s_RB_results.csv' % (dir_name, current_drug))
#######################

