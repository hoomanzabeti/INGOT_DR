# INGOT-DR

**INGOT-DR** ( **IN**terpretable **G**r**O**up **T**esting for **D**rug **R**esistance) is an interpretable rule-based
predictive model base on **Group Testing** and **Boolean Compressed Sesing**. For more details and citation please see
the [paper](#paper).

## Install
INGOT-DR can be installed from [PyPI](https://pypi.org/project/ingotdr/). 
```shell
pip install ingotdr
```
## Train and evaluate an INGOT-DR classifier

### Arguments

```python
ingot.INGOTClassifier( w_weight=1, lambda_p=1, lambda_z=1, lambda_e=1, false_positive_rate_upper_bound=None,
                       false_negative_rate_upper_bound=None, max_rule_size=None, rounding_threshold=1e-5,
                       lp_relaxation=False, only_slack_lp_relaxation=False, lp_rounding_threshold=0,
                       is_it_noiseless=False, solver_name='PULP_CBC_CMD', solver_options=None)
```

|Name|Type|Description|Default|
|:---|:---:|:---|:---:|
|w_weight|vector, float|A vector, float to provide prior weight to _w_.  | 1.0 |
|lambda_p| float| Regularization coefficient for positive labels.|1.0|
|lambda_z| float| Regularization coefficient for negative/zero labels.|1.0|
|lambda_e| float| Regularization coefficient for all slack variables.|1.0|
|false_positive_rate_upper_bound| float| False positive rate (FPR) upper bound.| None|
|false_negative_rate_upper_bound| float| False negative rate(FNR) upper bound.| None|
|max_rule_size| int | Maximum rule size.| None |
|rounding_threshold| float| Threshold for ILP solutions for Rounding to 0 and 1.| 1e-5|
|lp_relaxation| bool | A flag to use the lp relaxed version.| False|
|only_slack_lp_relaxation| bool| A flag to only use the lp relaxed slack variables.| False|
|lp_rounding_threshold| float| Threshold for lp solutions for Rounding to 0 and 1. Range from 0 to 1.| 0.0 |
|is_it_noiseless| bool| A flag to specify whether the problem is noisy or noiseless. |False|
|solver_name| str | Solver's [name](https://coin-or.github.io/pulp/guides/how_to_configure_solvers.html) provided by Pulp. | 'PULP_CBC_CMD' |
|solver_options| dict | Solver's [options](https://coin-or.github.io/pulp/technical/solvers.html) provided by Pulp.| None|

### Methods
|Method|Description|
|---|---|
|`fit(X,y)`|Fit the model with respect to the given data.|
|`get_params_dictionary(variable_type='w')`|Provide a dictionary of individuals with their status obtained by decoder. Type of the variable.e.g. 'w', 'ep' or 'en'|
|`solution()`|Provide a vector of binary features importance. i.e. 1 if feature was used in the model 0 otherwise.|
|`predict(X)`|Provide a predicted labels for X.|
|`score(X,y)`|Provide the accuracy of `self.predict(X)` with respect to `y`|
|`learned_rule(return_type='feature_name')`|Return a list of rules. return_type can be 'feature_name' or 'feature_id'.|
|`write(fileType='mps', **kwargs)`|Create a file from the problem. `fileType` can be 'mps', 'lp', 'json' or 'display'. 'display' shows the ILP/LP problem on screen.|

**Example:**
Sample data in the following example is available [here](https://github.com/hoomanzabeti/INGOT_DR_project/tree/master/data).
```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import ingot

feature_matrix = 'SNPsMatrix_ciprofloxacin.csv'
label_vector =  'ciprofloxacinLabel.csv'

X = pd.read_csv(feature_matrix, index_col=0)
y = pd.read_csv(label_vector, index_col=0).to_numpy().ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33, test_size=0.2, stratify=y)

clf = ingot.INGOTClassifier(lambda_p=10, lambda_z=0.01, false_positive_rate_upper_bound=0.1,
                            max_rule_size=20, solver_name='CPLEX_PY')
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print("Balanced accuracy: {}".format(balanced_accuracy_score(y_test, y_pred)))
print("Accuracy: {}".format(clf.score(X_test,y_test)))
print("Features in the learned rule: {}".format(clf.learned_rule()))
```
Output:

```shell
Balanced accuracy: 0.8449477351916377
Accuracy: 0.9550561797752809
Features in the learned rule: ['7570, C, T', '7572, T, C', '7581, G, T', '7582, A, C', '7582, A, G']
```

### Hyper-parameter tuning

**Example:**
```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import ingot
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

feature_matrix = 'SNPsMatrix_ciprofloxacin.csv'
label_vector =  'ciprofloxacinLabel.csv'

X = pd.read_csv(feature_matrix, index_col=0)
y = pd.read_csv(label_vector, index_col=0).to_numpy().ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33, test_size=0.2, stratify=y)

clf = ingot.INGOTClassifier(false_positive_rate_upper_bound=0.1, max_rule_size=20, solver_name='CPLEX_PY',
                            solver_options={'timeLimit': 1800})

scoring = dict(Accuracy='accuracy', balanced_accuracy=make_scorer(balanced_accuracy_score))
param_grid={'lambda_p': [ 1, 10, 100 ], 'lambda_z': [ 0.01, 0.1, 1 ]}
grid = GridSearchCV(estimator=clf, param_grid= param_grid, scoring=scoring, cv=5, refit ='balanced_accuracy',
                    n_jobs=-1, verbose= 3)
grid.fit(X_train, y_train)

y_pred = grid.predict(X_test)

print("Balanced accuracy: {}".format(balanced_accuracy_score(y_test, y_pred)))
print('Best params: {}'.format(grid.best_params_))
```
Output:

```shell
Balanced accuracy: 0.8449477351916377
Best params: {'lambda_p': 10, 'lambda_z': 0.01}
```


**Note:** _w_weight_ and _lambda_e_ are not part of the main ILP ([Eq (11)](#paper)) defined in the INGOT-DR paper. These two variables
 are defined to provide freedom when _Optimizing for different target metric_ ([section 1.4](#paper)) is needed. The 
complete objective function with these two variables would be:

![complete objective function](https://github.com/hoomanzabeti/INGOT_DR_project/blob/master/data/CompleteObjFunc.gif)

**Example:** 
Classifier corresponding to [Eq (15)](#paper) with maximum rule size k=20 and specificity lower bound t= 90% can be defined as following:
```python
clf = ingot.INGOTClassifier(w_weight=0, lambda_z=0, false_positive_rate_upper_bound=0.1, max_rule_size=20,
                            solver_name='CPLEX_PY')
```

|lp_relaxation|only_slack_lp_relaxation|is_it_noiseless|Equation number in the [paper](#paper)|
|---|---|---|:---:|
|False|False|False| Eq (11)|
|False|True|True|Eq (3)|
|False|True|False|Eq (4) with objective function of Eq (11)|
|False|False|True|Eq (3)|
|True|True|False|LP relaxation of Eq (4) with objective function of Eq (11)|
|True|False|False|LP relaxation of Eq (4) with objective function of Eq (11)|
|True|False|True|LP relaxation of Eq (3)|
True|True|True|LP relaxation of Eq (3)|


**Note:** True value of _lp_relaxation_ or _is_it_noiseless_ with override _only_slack_lp_relaxation_. i.e. if one of them is True
then value of _only_slack_lp_relaxation_ is not important.

### Solver 
INGOT-DR supports a variety of solvers through the [PuLP](https://pypi.org/project/PuLP/) application programming interface (API). 
Solvers such as [GLPK](http://www.gnu.org/software/glpk/glpk.html),
[COIN-OR CLP/CBC](https://github.com/coin-or/Cbc),
[CPLEX](http://www.cplex.com/),
[GUROBI](http://www.gurobi.com/),
[MOSEK](https://www.mosek.com/),
[XPRESS](https://www.fico.com/es/products/fico-xpress-solver),
[CHOCO](https://choco-solver.org/),
[MIPCL](http://mipcl-cpp.appspot.com/),
[SCIP](https://www.scipopt.org/).

List of available solvers on your machine:
```python
import pulp as pl
solver_list = pl.listSolvers(onlyAvailable=True)
```

Name and properties of the solver can be specified via ```solver_name``` and 
```solver_options```. e.g:
```python
clf = ingot.INGOTClassifier(solver_name='CPLEX_PY', solver_options={'timeLimit': 1800})
```
In the [INGOT-DR](#paper) paper, ```'CPLEX_PY'``` is the main solver. IBM CPLEX for academic use is available
[here](https://www.ibm.com/academic/technology/data-science). 
## Paper:

[INGOT-DR: an interpretable classifier forpredicting drug resistance in M. tuberculosis](https://www.biorxiv.org/content/10.1101/2020.05.31.115741v2.full).
([bibtex](https://scholar.googleusercontent.com/scholar.bib?q=info:bQ6FP1AQpvkJ:scholar.google.com/&output=citation&scisdr=CgXpW1OOEIO721E6sjI:AAGBfm0AAAAAYKw_qjKcmF8c1XZV57JWSMoDkwpaXPr8&scisig=AAGBfm0AAAAAYKw_qrApE1nCy1ns_BxQVZG_vrbY2Ot3&scisf=4&ct=citation&cd=-1&hl=en))
