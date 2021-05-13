import pulp as pl
import os
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score

from group_testing.generate_test_results import gen_test_vector
import group_testing.utils as utils


class INGOTClassifier(BaseEstimator, ClassifierMixin):
    """
    A class to represent INGOT-DR classifier
    """

    def __init__(self, w_weight=1, lambda_p=1, lambda_n=1, lambda_e=1, false_positive_rate_lower_bound=None,
                 false_negative_rate_lower_bound=None, max_rule_size=None, rounding_threshold=1e-5,
                 defective_num_lower_bound=None,
                 lp_relaxation=False, only_slack_lp_relaxation=False, lp_rounding_threshold=0,
                 is_it_noiseless=True, solver_name=None, solver_options=None):
        """
        Constructs all the necessary attributes for the decoder object.
        Parameters:
            w_weight (vector, int, float): A vector to provide prior weight. Default to vector of ones.
            lambda_p (int): Regularization coefficient for positive labels. Default to 1.
            lambda_n (int): Regularization coefficient for negative labels. Default to 1.
            lambda_e (int): Regularization coefficient for slack variables. Default to 1.
            false_positive_rate_lower_bound (float): False positive rate(FPR) lower bound. Default to None.
            false_negative_rate_lower_bound (float): False negative rate(FNR) lower bound. Default to None.
            max_rule_size (int): Maximum rule size. Default to None.
            rounding_threshold (float): Threshold for ilp solutions for Rounding to 0 and 1. Default to 1e-5
            defective_num_lower_bound (int): lower bound for number of infected people. Default to None.
            lp_relaxation (bool): A flag to use the lp relaxed version. Default to False.
            only_slack_lp_relaxation (bool): A flag to only use the lp relaxed slack variables. Default to False.
            lp_rounding_threshold (float): Threshold for lp solutions for Rounding to 0 and 1. Default to 0.
            Range from 0 to 1.
            is_it_noiseless (bool): A flag to specify whether the problem is noisy or noiseless. Default to True.
            solver_name (str): Solver's name provided by Pulp. Default to None.
            solver_options (dic): Solver's options provided by Pulp. Default to None.
        """

        self.lambda_e = lambda_e
        self.lambda_p = lambda_p
        self.lambda_n = lambda_n
        try:
            assert isinstance(w_weight, (int, float, list))
            self.w_weight = w_weight
        except AssertionError:
            print("w_weight should be either int, float or list of numbers")
        self.false_positive_rate_lower_bound = false_positive_rate_lower_bound
        self.false_negative_rate_lower_bound = false_negative_rate_lower_bound
        self.max_rule_size = max_rule_size
        self.rounding_threshold = rounding_threshold
        self.defective_num_lower_bound = defective_num_lower_bound
        self.lp_relaxation = lp_relaxation
        self.only_slack_lp_relaxation = only_slack_lp_relaxation
        self.lp_rounding_threshold = lp_rounding_threshold
        self.is_it_noiseless = is_it_noiseless
        self.solver_name = solver_name
        self.solver_options = solver_options
        self.prob_ = None

    def fit(self, A, label):
        """
        Function to a decode base of design matrix and test results
        Parameters:
            A (binary numpy 2d-array): The group testing matrix.
            label (binary numpy array): The vector of results of the group tests.
        Returns:
            self (GroupTestingDecoder): A decoder object including decoding solution
        """
        m, n = A.shape
        alpha = A.sum(axis=1)
        label = np.array(label)
        positive_label = np.where(label == 1)[0]
        negative_label = np.where(label == 0)[0]
        if self.false_positive_rate_lower_bound is not None:
            self.false_positive_lower_bound = self.false_positive_rate_lower_bound * len(negative_label)
        if self.false_negative_rate_lower_bound is not None:
            self.false_negative_lower_bound = self.false_negative_rate_lower_bound * len(positive_label)
        # -------------------------------------
        # Checking length of w_weight
        try:
            if isinstance(self.w_weight, list):
                assert len(self.w_weight) == n
        except AssertionError:
            print("length of w_weight should be equal to number of individuals( numbers of columns in the group "
                  "testing matrix)")
        # -------------------------------------
        # Initializing the ILP problem
        p = pl.LpProblem('GroupTesting', pl.LpMinimize)

        # Variables kind
        if self.lp_relaxation:
            wVarCategory = 'Continuous'
        else:
            wVarCategory = 'Binary'
        # Variable w
        w = pl.LpVariable.dicts('w', range(n), lowBound=0, upBound=1, cat=wVarCategory)
        # --------------------------------------
        # Noiseless setting
        if self.is_it_noiseless:
            # Defining the objective function
            p += pl.lpSum([self.w_weight * w[i] if isinstance(self.w_weight, (int, float)) else self.w_weight[i] * w[i]
                           for i in range(n)])
            # Constraints
            for i in positive_label:
                p += pl.lpSum([A[i][j] * w[j] for j in range(n)]) >= 1
            for i in negative_label:
                p += pl.lpSum([A[i][j] * w[j] for j in range(n)]) == 0
        # --------------------------------------
        # Noisy setting
        else:
            if self.lp_relaxation or self.only_slack_lp_relaxation:
                en_upBound = None
                varCategory = 'Continuous'
            else:
                en_upBound = 1
                varCategory = 'Binary'
            # Variable ep
            if len(positive_label) != 0:
                ep = pl.LpVariable.dicts(name='ep', indexs=list(positive_label), lowBound=0, upBound=1, cat=varCategory)
            else:
                ep = []
            # Variable en
            if len(negative_label) != 0:
                en = pl.LpVariable.dicts(name='en', indexs=list(negative_label), lowBound=0, upBound=en_upBound,
                                         cat=varCategory)
            else:
                en = []
            # Defining the objective function
            p += pl.lpSum([self.w_weight * w[i] if isinstance(self.w_weight, (int, float)) else self.w_weight[i] * w[i]
                           for i in range(n)]) + \
                 pl.lpSum([self.lambda_e * self.lambda_p * ep[j] for j in positive_label]) + \
                 pl.lpSum([self.lambda_e * self.lambda_n * en[k] for k in negative_label])
            # Constraints
            for i in positive_label:
                p += pl.lpSum([A[i][j] * w[j] for j in range(n)] + ep[i]) >= 1
            for i in negative_label:
                if varCategory == 'Continuous':
                    p += pl.lpSum([A[i][j] * w[j] for j in range(n)] + -1 * en[i]) == 0
                else:
                    p += pl.lpSum([-1 * A[i][j] * w[j] for j in range(n)] + alpha[i] * en[i]) >= 0
            # Additional constraints
            if (self.max_rule_size is not None) and (not self.lp_relaxation):
                p += pl.lpSum([w[i] for i in range(n)]) <= self.max_rule_size
            if self.false_negative_rate_lower_bound is not None and not self.lp_relaxation and \
                    not self.only_slack_lp_relaxation and len(ep) != 0:
                p += pl.lpSum(ep) <= self.false_negative_lower_bound
            if self.false_positive_rate_lower_bound is not None and not self.lp_relaxation and \
                    not self.only_slack_lp_relaxation and len(en) != 0:
                p += pl.lpSum(en) <= self.false_positive_lower_bound

        if self.solver_options is not None:
            solver = pl.get_solver(self.solver_name, **self.solver_options)
        else:
            solver = pl.get_solver(self.solver_name)
        p.solve(solver)
        if not self.lp_relaxation:
            p.roundSolution()
        # ----------------
        self.prob_ = p
        # print("Status:", pl.LpStatus[p.status])
        return self

    def get_params_w(self, variable_type='w'):
        """
        Function to provide a dictionary of individuals with their status obtained by decoder.
        Parameters:
            self (GroupTestingDecoder): Decoder object.
            variable_type (str): Type of the variable.e.g. 'w','ep' or 'en'
        Returns:
            w_solutions_dict (dict): A dictionary of individuals with their status.
        """
        try:
            assert self.prob_ is not None
            # w_solution_dict = dict([(v.name, v.varValue)
            # for v in self.prob_.variables() if variable_type in v.name and v.varValue > 0])
            # Pulp uses ASCII sort when we recover the solution. It would cause a lot of problems when we want
            # to use the solution. We need to use alphabetical sort based on variables names (v.names). To do so
            # we use utils.py and the following lines of codes
            w_solution_dict = dict([(v.name, v.varValue)
                                    for v in self.prob_.variables() if variable_type in v.name])
            index_map = {v: i for i, v in enumerate(sorted(w_solution_dict.keys(), key=utils.natural_keys))}
            w_solution_dict = {k: v for k, v in sorted(w_solution_dict.items(), key=lambda pair: index_map[pair[0]])}
        except AttributeError:
            raise RuntimeError("You must fit the data first!")
        return w_solution_dict

    def solution(self):
        """
        Function to provide a vector of decoder solution.
        Parameters:
            self (GroupTestingDecoder): Decoder object.
        Returns:
            w_solutions (vector): A vector of decoder solution.
        """
        try:
            assert self.prob_ is not None
            # w_solution = [v.name[2:] for v in self.prob_.variables() if v.name[0] == 'w' and v.varValue > 0]
            # Pulp uses ASCII sort when we recover the solution. It would cause a lot of problems when we want
            # to use the solution. We need to use alphabetical sort based on variables names (v.names). To do so
            # we use utils.py and the following lines of codes
            w_solution = self.get_params_w()
            index_map = {v: i for i, v in enumerate(sorted(w_solution.keys(), key=utils.natural_keys))}
            w_solution = [v for k, v in sorted(w_solution.items(), key=lambda pair: index_map[pair[0]])]
            if self.lp_relaxation:
                w_solution = [1 if i > self.lp_rounding_threshold else 0 for i in w_solution]
        except AttributeError:
            raise RuntimeError("You must fit the data first!")
        return w_solution

    def predict(self, A):
        """
        Function to predict test results based on solution.
        Parameters:
            self (INGOTClassifier): Classifier object.
            A (binary numpy 2d-array): The feature array or matrix.
        Returns:
             A vector of predicted labels.
        """
        return np.minimum(np.matmul(A, self.solution()), 1)

    def write(self):
        pass
if __name__ == '__main__':
    clf = INGOTClassifier()
    A = np.random.randint(2, size=(2, 4))
    y = np.random.randint(2, size=2)
    INGOTClassifier.fit(A,y)
