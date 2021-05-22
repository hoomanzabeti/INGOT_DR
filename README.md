# INGOT-DR

**INGOT-DR** ( **IN**terpretable **G**r**O**up **T**esting for **D**rug **R**esistance) is an interpretable rule-based predictive model base on **Group Testing** and **Boolean Compressed Sesing**.

## Train and evaluate an INGOT-DR classifier
```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
from INGOT import INGOTClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33, test_size=0.2, stratify=y)

clf = INGOTClassifier()
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print("Balanced accuracy: {}".format(balanced_accuracy_score(y_test, y_pred)))


```
