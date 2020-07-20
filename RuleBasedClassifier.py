#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 14:57:49 2020

@author: hoomanzabeti
"""

import numpy as np
import copy
from sklearn.base import BaseEstimator, ClassifierMixin
import cplex
from sklearn.metrics import f1_score

class drugClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, rc=1, rc_e=1, rc_p=0.1, rc_z=0.1, cpx_timelimit=600,\
                 rule_max_length=None, FN_up=None, FP_up=None):

        self.rc = rc
        self.rc_e = rc_e
        self.rc_p = rc_p
        self.rc_z = rc_z
        self.cpx_timelimit = cpx_timelimit
        self.rule_max_length = rule_max_length
        self.FN_up = FN_up
        self.FP_up = FP_up
        self.FP_up_percent = FP_up

    def fit(self, X, y):

        m, n = X.shape
        row_sum = list(X.sum(axis=1))
        self.m_ = m
        self.n_ = n
        X = X.values.tolist()
        # Positive and zero split                                                                                                                                                                                                                                                                                                                                                                                                                                                      
        P = [i for i in range(len(y)) if y[i] == 1]
        Z = [i for i in range(len(y)) if i not in P]
        # FN_up and FP_up are in percentage. We should convert them to get the
        # actual numbers
        if self.FN_up is not None:
            self.FN_up = self.FN_up * len(P)
        if self.FP_up is not None:
            self.FP_up  = self.FP_up * len(Z)
        # cplex IP problem setup                                                                                                                                                                                                                                                                                                                                                                                                                                                       
        p_obj = [self.rc] * n + [self.rc_e * self.rc_p] * len(P) + [self.rc_e * self.rc_z] * len(Z)
        p_lb = [0] * n + [0] * m
        p_ub = [1] * n + [1] * m
        p_ctype = ["B"] * n + ["B"] * m
        p_colnames = ["w" + str(i) for i in range(n)] + ["ep" + str(j) for j in P] + ["ez" + str(j) for j in Z]

        additional_rows = [x for x in [self.rule_max_length, self.FN_up, self.FP_up] if x is not None]
        p_rownames = ["r" + str(i) for i in range(m + len(additional_rows))]
        p_rhs = [1] * len(P) + [0] * len(Z) + additional_rows
        p_sense = ["G"] * len(P) + ["G"] * len(Z) + ["L"] * len(additional_rows)

        rows = []
        prob = cplex.Cplex()
        prob.parameters.timelimit.set(self.cpx_timelimit)
        # --------------------------------------                                                                                                                                                                                                                                                                                                                                                                                                                                       
        prob.objective.set_sense(prob.objective.sense.minimize)
        prob.variables.add(obj=p_obj, lb=p_lb, ub=p_ub, types=p_ctype, names=p_colnames)
        for i in P:
            row_P_name = ["w" + str(j) for j in range(n) if X[i][j] != 0] + ["ep" + str(i)]
            row_P_value = [X[i][j] for j in range(n) if X[i][j] != 0] + [1]
            rows.append(copy.copy([row_P_name, row_P_value]))

        for i in Z:
            row_Z_name = ["w" + str(j) for j in range(n) if X[i][j] != 0] + ["ez" + str(i)]
            row_Z_value = [-X[i][j] for j in range(n) if X[i][j] != 0] + [row_sum[i]]
            rows.append(copy.copy([row_Z_name, row_Z_value]))
        # ----------------- Additional constraints -----------------                                                                                                                                                                                                                                                                                                                                                                                                                   
        if self.rule_max_length is not None:
            row_w_name = ["w" + str(j) for j in range(n)]
            row_w_value = [1 for j in range(n)]
            rows.append(copy.copy([row_w_name, row_w_value]))
        if self.FN_up is not None:
            row_ep_name = ["ep" + str(j) for j in P]
            row_ep_value = [1 for j in P]
            rows.append(copy.copy([row_ep_name, row_ep_value]))
        if self.FP_up is not None:
            row_ez_name = ["ez" + str(j) for j in Z]
            row_ez_value = [1 for j in Z]
            rows.append(copy.copy([row_ez_name, row_ez_value]))
        # ----------------------------------------------------------                                                                                                                                                                                                                                                                                                                                                                                                                   
        prob.linear_constraints.add(lin_expr=rows, senses=p_sense, rhs=p_rhs, names=p_rownames)
        #prob.set_log_stream(None)
        #prob.set_error_stream(None)
        #prob.set_warning_stream(None)
        #prob.set_results_stream(None)
        prob.solve()
        self.prob_obj_ = prob
        # Note that in the following line we should have prob.solution.get_values(v) != 0
        # However, since their might be small cal error we might have some numbers like 
        # "-1.2184445573610454e-13". Therefore we set prob.solution.get_values(v) >= 0.5. 
        self.w_solution_ = [int(v[1:]) for v in prob.variables.get_names() if
                            v[0] == 'w' and prob.solution.get_values(v) >= 0.5]
        print(self.w_solution_)
        print('------------------------------------------------')
        print(len(self.w_solution_))
        return self
    def predict(self, X, y=None):

        try:
            getattr(self, "w_solution_")
            w = [1 if i in self.w_solution_ else 0 for i in range(self.n_)]
            y_pred = [1 if i != 0 else 0 for i in np.dot(X.values.tolist(), w)]
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        return y_pred

    def score(self, X, y=None, metric='F1'):
        if metric == 'F1':
            score_value = f1_score(y, self.predict(X))
        return score_value
