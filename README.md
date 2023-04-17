## ImbalanceMetrics

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Description
The "ImbalanceMetrics" Python package provides an extensive set of evaluation metrics designed for assessing the performance of machine learning models on imbalanced datasets, where traditional accuracy and error rate measurements may be inadequate. The package incorporates several evaluation metrics that tackle the challenges specific to imbalanced domains, offering a more accurate evaluation of model performance.
<br>

## Requirements
1. Python 3
2. scikit-learn
3. NumPy
4. Pandas
5. Smogn

## Installation
```python
## install pypi release
pip install imbalance-metrics

## install developer version
pip install git+https://github.com/paobranco/ImbalanceMetrics.git
```

## Usage (Classification)
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imbalance_metrics import classification_metrics as cm
df = pd.read_csv('glass0.csv', header=None)
X,y=df.drop(columns=[9]),df[9]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
y_proba=clf.predict_proba(X_test)
y_phi = cm.calculate_classification_phi(y_test) # Relevance by class frequency
gmean = cm.gmean_score(y_test, y_pred) # Weighted gmean 
p_d0,r_d0,pra_d0=cm.pr_davis(y_test,y_proba,True) # Default minority as positive
p_d1,r_d1,pra_d1=cm.pr_davis(y_test,y_proba,True,pos_label=1) # 1 as positive
p_m0,r_m0,pra_m0=cm.pr_manning(y_test,y_proba,True) # Default minority as positive
p_m1,r_m1,pra_m1=cm.pr_manning(y_test,y_proba,True,pos_label=1) # 1 as positive 
cv_davis=cm.cross_validate_auc(clf,X,y,cm.pr_davis,5)
cv_manning=cm.cross_validate_auc(clf,X,y,cm.pr_manning,5)
```

## Usage (Regression)
```python
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imbalance_metrics import regression_metrics as rm
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
```

## Contributions

ImablanceMetrics is open for improvements and maintenance. Your help is valued to make the package better for everyone.

## License

Licensed under the General Public License v3.0 (GPLv3).

## Reference

- Ribeiro, R.P.: Utility-based regression. Ph.D. thesis, Dep. Computer Science, Faculty of Sciences - University of Porto (2011)
- Branco, P., Ribeiro, R.P., Torgo, L.: UBL: an R package for utility-based learning (2016), https://arxiv.org/abs/1604.08079
- Branco, P., Torgo, L., Ribeiro, R.: A survey of predictive modelling under imbalanced distributions (2015)
- Branco, P., Torgo, L., Ribeiro, R.P.: A survey of predictive modeling on imbalanced domains. ACM Computing Surveys (CSUR) 49(2), 1–50 (2016)
- Branco, P., Torgo, L., Ribeiro, R.P.: SMOGN: a pre-processing approach for imbalanced regression. In: First international workshop on learning with imbalanced domains: Theory and applications. pp. 36–50. PMLR (2017)
- Cordón, I., García, S., Fernández, A., Herrera, F.: Imbalance: Oversampling algorithms for imbalanced classification in r. Knowledge-Based Systems 161, 329–341 (2018), https://doi.org/10.1016/j.knosys.2018.07.035
- Davis, J., Goadrich, M.: The relationship between precision-recall and roc curves. vol. 06 (06 2006). https://doi.org/10.1145/1143844.1143874
- Derrac, J., Garcia, S., Sanchez, L., Herrera, F.: Keel data-mining software tool: Data set repository, integration of algorithms and experimental analysis framework. J. Mult. Valued Logic Soft Comput 17 (2015)
- Gaudreault, J.G., Branco, P., Gama, J.: An analysis of performance metrics for imbalanced classification. In: Discovery Science: 24th International Conference, DS 2021, Halifax, NS, Canada, October 11–13, 2021, Proceedings. pp. 67–77 (2021)
- Kubat, M., Matwin, S., et al.: Addressing the curse of imbalanced training sets: one-sided selection. In: Icml. vol. 97, p. 179. Citeseer (1997)
- Kunz, N.: SMOGN: Synthetic minority over-sampling technique for regression with gaussian noise (2020), https://pypi.org/project/smogn
- Lemaître, G., Nogueira, F., Aridas, C.K.: Imbalanced-learn: A python toolbox to tackle the curse of imbalanced datasets in machine learning. JMLR 18(17), 1–5 (2017),  http://jmlr.org/papers/v18/16-365.html
- Ribeiro, R.: Utility-based Regression. Ph.D. thesis, Dep. Computer Science, Faculty of Sciences - University of Porto (2011)
- Ribeiro, R., Moniz, N.: Imbalanced regression and extreme value prediction. Machine Learning 109,1–33 (09 2020). https://doi.org/10.1007/s10994-020-05900-9
- berreergun: Ironpy. https://github.com/berreergun/IRonPy (2021)
