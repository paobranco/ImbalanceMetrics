.. imbalanced_metrics documentation master file, created by
   sphinx-quickstart on Mon Feb  6 21:50:15 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to imbalanced_metrics's documentation!
==============================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Installation
------------

.. code-block:: bash

   pip install imbalanced-metrics

Usage Classification 
-----

.. code-block:: python

   import pandas as pd
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split
   from imbalanced_metrics import classification_metrics as cm
   df = pd.read_csv('glass0.csv', header=None)
   X,y=df.drop(columns=[9]),df[9]
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40)
   clf = RandomForestClassifier(random_state=42)
   clf.fit(X_train,y_train)
   y_pred=clf.predict(X_test)
   y_proba=clf.predict_proba(X_test)
   gmean = cm.gmean_score(y_test, y_pred)
   p_d0,r_d0,pra_d0=cm.pr_davis(y_test,y_proba,True) # Default minority as positive
   p_d1,r_d1,pra_d1=cm.pr_davis(y_test,y_proba,True,pos_label=1) # 1 as positive
   p_m0,r_m0,pra_m0=cm.pr_manning(y_test,y_proba,True) # Default minority as positive
   p_m1,r_m1,pra_m1=cm.pr_manning(y_test,y_proba,True,pos_label=1) # 1 as positive 
   cv_davis=cm.cross_validate_auc(clf,X,y,cm.pr_davis,5)
   cv_manning=cm.cross_validate_auc(clf,X,y,cm.pr_manning,5)


Usage Regression
-----

.. code-block:: python
   
   import pandas as pd
   from sklearn.svm import SVR
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler
   from imbalanced_metrics import regression_metrics as rm
   df = pd.read_csv('housing(processed).csv')
   X,y=df.drop(columns="SalePrice"),df["SalePrice"]
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
   sc = StandardScaler()
   y_train = sc.fit_transform(y_train.values.reshape(-1, 1))
   y_test = sc.transform (y_test.values.reshape(-1, 1))
   reg = SVR().fit(X_train, y_train)
   y_pred = reg.predict(X_test)
   y_test=y_test.reshape(-1)
   y_pred=y_pred.reshape(-1)
   wmse = rm.phi_weighted_mse (y_test , y_pred)
   wmae = rm.phi_weighted_mae (y_test , y_pred)
   wr2 = rm.phi_weighted_r2 (y_test , y_pred)
   wrmse = rm.phi_weighted_root_mse (y_test , y_pred) 
   ser_t = rm.ser_t(y_test,y_pred,t=.7)
   sera= rm.sera(y_test,y_pred,return_err = True)

.. toctree::
   :caption: api_classification

   api_classification

.. toctree::
   :caption: api_regression

   api_regression


