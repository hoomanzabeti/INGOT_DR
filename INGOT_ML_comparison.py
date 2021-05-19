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
import os
import time
import argparse
import utils
import INGOT
import numpy as np


def main(sysargs=sys.argv[1:]):
    parser = argparse.ArgumentParser(prog='INGOT-DR', description='Description')
    parser.add_argument(
        '--config', dest='config', metavar='FILE',
        help='Path to the config file', required=False, default='config.yml'
    )
    parser.add_argument(
        '--output-dir', dest='output_path', metavar='DIR',
        help='Path to the output directory'
    )
    parser.add_argument(
        '--data-file', dest='data_file', metavar='FILE',
        help='Path to the input data in csv', required=False, default='SNPsMatrix_ciprofloxacin.csv'
    )
    parser.add_argument(
        '--label-file', dest='label_file', metavar='FILE',
        help='Path to the label file in csv', required=False, default='ciprofloxacinLabel.csv'
    )
    parser.add_argument(
        '--drug-name', dest='current_drug', metavar='FILE',
        help='Path to the label file in csv', required=False, default='ciprofloxacin'
    )
    parser.add_argument(
        '--model-name', dest='current_model', metavar='FILE',
        help='Path to the label file in csv', required=False, default='INGOT'
    )
    args = parser.parse_args()
    current_drug = args.current_drug
    current_model = args.current_model
    print_val = lambda p: print('{} {} {}'.format('-' * (50 - len(p) // 2), p, '-' * (50 - len(p) // 2)))
    print_val(current_drug)
    print_val(current_model)
    current_path, result_path = utils.result_path_generator(args)
    config_file = utils.config_reader(args.config)
    print_val('Loading Data')
    X = pd.read_csv(args.data_file, index_col=0)
    print('Data file shape: {}'.format(X.shape))
    y = pd.read_csv(args.label_file, index_col=0).to_numpy().ravel()
    print('Label file shape: {}'.format(y.shape))
    print('Label ratio: R: {}, S: {}'.format(len(np.where(y == 1)[0]), len(np.where(y == 0)[0])))
    print_val('Train-Test split')
    for i in config_file['TrainTestSplit'].items(): print('{}: {}'.format(i[0], i[1]))
    X_train, X_test, y_train, y_test = train_test_split(X, y, **config_file['TrainTestSplit'], stratify=y)
    print('Train data shape: {}'.format(X_train.shape))
    print('Train label shape: {}'.format(y_train.shape))
    print('Test data shape: {}'.format(X_test.shape))
    print('Test label shape: {}'.format(y_test.shape))
    scoring = dict(Accuracy='accuracy', tp=make_scorer(utils.tp), tn=make_scorer(utils.tn),
                   fp=make_scorer(utils.fp), fn=make_scorer(utils.fn),
                   balanced_accuracy=make_scorer(balanced_accuracy_score))
    print_val('Hyper-parameter Tuning')
    for i in config_file['CrossValidation'].items(): print('{}: {}'.format(i[0], i[1]))
    current_module = utils.my_import(config_file['Models'][args.current_model]['module'])
    dClassifier = getattr(current_module, config_file['Models'][args.current_model]['model'])
    dClassifier = dClassifier(**config_file['Models'][current_model]['params'])
    grid = GridSearchCV(estimator=dClassifier, param_grid=config_file['Models'][current_model]['cv'],
                        scoring=scoring, **config_file['CrossValidation'])
    print_val('Training')
    start_time = time.time()
    grid.fit(X_train, y_train)
    end_time = time.time()
    print_val('Saving Cross-Validation Result')
    print('Best params: {}'.format(grid.best_params_))
    pd.DataFrame.from_dict(grid.cv_results_).to_csv(os.path.join(result_path,
                                                                 '{}_{}_GridSearchCV.csv'.format(current_drug,
                                                                                                 current_model)))
    pd.DataFrame(grid.best_params_, index=[0]).to_csv(os.path.join(result_path,
                                                                   '{}_{}_BestParameter.csv'.format(current_drug,
                                                                                                    current_model)))
    if not isinstance(grid.best_estimator_, INGOT.INGOTClassifier):
        dump(grid.best_estimator_, os.path.join(result_path,
                                                '{}_{}_BestModel.joblib'.format(current_drug, current_model)))
    print_val('Validation')
    y_pred = grid.predict(X_test)
    if utils.dict_key_checker(config_file['Models'][current_model], 'shap_report'):
        if config_file['Models'][current_model]['shap_report']:
            feature_importance = utils.shap_vals(grid.best_estimator_, X_train, X_test,
                                                 config_file['Models'][current_model]['shap_kernel'])
            feature_importance.to_csv(os.path.join(result_path,
                                                   '{}_{}_shap_feature_importance.csv'.format(current_drug,
                                                                                              current_model)))
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

    print('{}: Balanced Accuracy: {} '.format(current_drug,
                                                                    balanced_accuracy_score(y_test, y_pred)))
    print_val('Saving Result')
    pd.DataFrame([test_results]).to_csv(os.path.join(result_path,
                                                     '{}_{}_results.csv'.format(current_drug, current_model)))
    for i in test_results.items(): print('{}: {}'.format(i[0], i[1]))
    print_val('DONE')
if __name__ == "__main__":
    main()
#######################
