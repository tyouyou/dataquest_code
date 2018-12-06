## 2. Combining Model Predictions With Ensembles ##

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

columns = ["age", "workclass", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "hours_per_week", "native_country"]

clf = DecisionTreeClassifier(random_state=1, min_samples_leaf=2)
clf.fit(train[columns], train["high_income"])

clf2 = DecisionTreeClassifier(random_state=1, max_depth=5)
clf2.fit(train[columns], train["high_income"])

pre1 = clf.predict(test[columns])
pre2 = clf2.predict(test[columns])

auc1 = roc_auc_score(test['high_income'], pre1)
auc2 = roc_auc_score(test['high_income'], pre2)

print(auc1)
print(auc2)


## 4. Combining Our Predictions ##

predictions = clf.predict_proba(test[columns])[:,1]
predictions2 = clf2.predict_proba(test[columns])[:,1]

pred = (predictions + predictions2)/2
rounded = numpy.round(pred)
auc_pred = roc_auc_score(test['high_income'], rounded)
print(auc_pred)
