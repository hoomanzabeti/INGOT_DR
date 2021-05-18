#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 14:56:07 2020

@author: hoomanzabeti
"""

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
from joblib import dump
import sys
import datetime
import os
import time
import argparse
from INGOT import INGOTClassifier
import utils
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
rc_p_list = [1e-2, 1e-1, 1, 1e1, 1e2]
rc_z_list = [1e-2, 1e-1, 1, 1e1, 1e2]
cplex_time_limit = [1800]


# Loading  the data for drug 'current_drug'


def main(sysargs=sys.argv[1:]):
    parser = argparse.ArgumentParser(prog='GroupTesting', description='Description')
    parser.add_argument(
        '--config', dest='config', metavar='FILE',
        help='Path to the config file', required=True
    )
    parser.add_argument(
        '--output-dir', dest='output_path', metavar='DIR',
        help='Path to the output directory'
    )
    parser.add_argument(
        '--data-file', dest='data_file', metavar='FILE',
        help='Path to the input data in csv', required=True
    )
    parser.add_argument(
        '--label-file', dest='label_file', metavar='FILE',
        help='Path to the label file in csv', required=True
    )
    parser.add_argument(
        '--drug-name', dest='current_drug', metavar='FILE',
        help='Path to the label file in csv', required=True
    )
    parser.add_argument(
        '--model-name', dest='current_model', metavar='FILE',
        help='Path to the label file in csv', required=True
    )
    args = parser.parse_args()
    current_drug = args.current_drug
    current_model = args.current_model
    current_path, result_path = utils.result_path_generator(args)
    config_file = utils.config_reader(args.config)
    print('----------------------Data Loading -------------------------------')
    X = pd.read_csv(args.data_file, index_col=0)
    y = pd.read_csv(args.label_file, index_col=0).to_numpy().ravel()
    snp_list = list(X.columns)
    print('----------------------Train-Test split-------------------------------')
    X_train, X_test, y_train, y_test = train_test_split(X, y, **config_file['TrainTestSplit'])

    scoring = dict(Accuracy='accuracy', tp=make_scorer(utils.tp), tn=make_scorer(utils.tn),
                   fp=make_scorer(utils.fp), fn=make_scorer(utils.fn),
                   balanced_accuracy=make_scorer(balanced_accuracy_score))

    print('----------------------Hyper-parameter Tuning-------------------------------')
    current_module = utils.my_import(config_file['Models'][args.current_model]['module'])
    dClassifier = getattr(current_module, config_file['Models'][args.current_model]['model'])
    grid = GridSearchCV(estimator=dClassifier, param_grid=config_file['Models'][current_model]['params'],
                        scoring=scoring, **config_file['CrossValidation'])
    print('----------------------Training-------------------------------')
    start_time = time.time()
    grid.fit(X_train, y_train)
    end_time = time.time()
    print('----------------------Saving Cross-Validation Result-------------------------------')
    pd.DataFrame.from_dict(grid.cv_results_).to_csv(os.path.join(result_path,
                                                                 '{}_{}_GridSearchCV.csv'.format(current_drug,
                                                                                                 current_model)))
    pd.DataFrame(grid.best_params_, index=[0]).to_csv(os.path.join(result_path,
                                                                   '{}_{}_BestParameter.csv'.format(current_drug,
                                                                                                    current_model)))
    dump(grid.best_estimator_, os.path.join(result_path, '{}_{}_BestModel.joblib'.format(current_drug, current_model)))
    print('----------------------Validation-------------------------------')
    y_pred = grid.predict(X_test)
    if config_file['Models'][current_model]['shap_report']:
        feature_importance = utils.shap_vals(grid.best_estimator_, X_train, X_test, current_model)
        feature_importance.to_csv(os.path.join(result_path,
                                               '{}_{}_shap_feature_importance.csv'.format(current_drug, current_model)))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    if current_model == 'INGOT':
        test_results = dict(Drug=current_drug, time=round(end_time - start_time, 2),
                            balanced_accuracy=balanced_accuracy_score(y_test, y_pred), tp=tp, tn=tn, fp=fp, fn=fn,
                            sensitivity=tp / (tp + fn), specificity=tn / (tn + fp),
                            FPR_upper_bound=grid.best_estimator_.false_positive_rate_upper_bound,
                            Max_rule_size=grid.best_estimator_.max_rule_size,
                            Rule_size=len(grid.best_estimator_.learned_rule(return_type='feature_id')),
                            rule=grid.best_estimator_.learned_rule(return_type='feature_id'),
                            SNP=grid.best_estimator_.learned_rule(return_type='feature_name'))
    else:
        test_results = dict(Drug=current_drug, time=round(end_time - start_time, 2),
                            balanced_accuracy=balanced_accuracy_score(y_test, y_pred), tp=tp, tn=tn, fp=fp, fn=fn,
                            sensitivity=tp / (tp + fn), specificity=tn / (tn + fp))

    print('----------{}: Balanced Accuracy{}-----------'.format(current_drug,
                                                                balanced_accuracy_score(y_test, y_pred)))
    print('----------------------Saving Result-------------------------------')
    pd.DataFrame(test_results, index=[0]).to_csv(os.path.join(result_path,
                                                              '{}_{}_results.csv'.format(current_drug,
                                                                                         current_model)))


if __name__ == "__main__":
    main()
#######################
