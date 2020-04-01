# -*- coding: utf-8 -*-

# Tianchi secondhand car saleprice prediction
# -  Fourth part - Model developing and model tuning


# 4.1 import package and data
# --------------------------------------------------------------------------------------#
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import GridSearchCV
from bayes_opt import BayesianOptimization
from sklearn.model_selection import learning_curve

data = pd.read_csv('data.csv')

# 4.2 cross-validation of base model
# --------------------------------------------------------------------------------------#
# here choose XGB and LGB as the two base model
y_train_all = data['price']
x_train_all = data.drop(['price'], axis=1)

#
# # 4.2.1 Define the  n-fold cross valid function
def nfoldcross(n, model, x_train_all, y_train_all):
    scores_train = []
    scores = []
    kf = KFold(n_splits=n, shuffle=True, random_state=0)
    for train_ind, val_ind in kf.split(x_train_all, y_train_all):
        train_x = x_train_all.iloc[train_ind].values
        train_y = y_train_all.iloc[train_ind]
        val_x = x_train_all.iloc[val_ind].values
        val_y = y_train_all.iloc[val_ind]

        model.fit(train_x, train_y)
        pred_train_xgb = model.predict(train_x)
        pred_xgb = model.predict(val_x)

        score_train = mean_absolute_error(train_y, pred_train_xgb)
        scores_train.append(score_train)
        score = mean_absolute_error(val_y, pred_xgb)
        scores.append(score)

    print('Train mae:', np.mean(score_train))
    print('Val mae', np.mean(scores))


# 4.2.2 Build default XGB
xgr = xgb.XGBRegressor()
nfoldcross(5, xgr, x_train_all, y_train_all)

# print result
# Train mae: 538.1020471689657
# Val mae 651.9790244239116

# 4.2.3 Build default LGB
lgr = lgb.LGBMRegressor()
nfoldcross(5, lgr, x_train_all, y_train_all)

# print result
# Train mae: 657.4689273197503
# Val mae 691.1988738267048

# conclusion
# seems that xgb more likely to be overfitted

# 4.3 Model tuning
# --------------------------------------------------------------------------------------#

# 4.3.1 Gridsearch
# After Two round grid search  of XGB, determine the parameter
xgr = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, gamma=0, subsample=0.8)
parameters = {'max_depth': [3, 4, 5, 6, 7, 8, 9], 'colsample_bytree': np.linspace(0.4, 1, 7), 'gamma': [0, 0.2, 0.4]}

clf = GridSearchCV(xgr, parameters, cv=KFold(n_splits=5, shuffle=True, random_state=0))
clf.fit(x_train_all, y_train_all)

xgr = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, subsample=0.8, \
                       colsample_bytree=0.8, max_depth=9)
parameter_l1 = {'reg_alpha': [1, 10, 100, 1000, 10000], 'reg_lambda': [1, 10, 100, 1000, 10000],
                'gamma': [1, 10, 100, 1000, 10000]}

clf = GridSearchCV(xgr, parameter_l1, cv=KFold(n_splits=5, shuffle=True, random_state=0))
clf.fit(x_train_all, y_train_all)

# run optimized model again
xgr_opt = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, subsample=0.8,
                           colsample_bytree=0.8, max_depth=10, reg_alpha=100, reg_lambda=0,
                           gamma=10000)
nfoldcross(5, xgr_opt, x_train_all, y_train_all)

# print result
# Train mae: 340.93358686069945
# Val mae 560.5813937310413

# MAE improving from 651 to 560 for XGB


# grid search for lgb
lgr = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, colsample_bytree=0.8, subsample=0.7)
parameters = {'max_depth': [10, 12, 14, 16], 'num_leaves': list(range(180, 320, 40))}
clf = GridSearchCV(lgr, parameters, cv=KFold(n_splits=3, shuffle=True, random_state=0))
clf.fit(x_train_all, y_train_all)

lgr_opt = lgb.LGBMRegressor(num_leaves=180, n_estimators=100, learning_rate=0.1, max_depth=12,
                            colsample_bytree=0.8, subsample=0.7)
nfoldcross(5, lgr_opt, x_train_all, y_train_all)


# print result
# Train mae: 461.61276745651116
# # Val mae 576.0627450905897

# MAE improve from 691 to 576


# 4.3.3 BayesianOptimization

def rf_cv(gamma, reg_lambda, reg_alpha, max_depth, subsample, colsample_bytree):
    val = cross_val_score(
        xgb.XGBRegressor(
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            max_depth=int(max_depth),
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            gamma=gamma),
        X=x_train_all, y=y_train_all, verbose=0, cv=3, scoring=make_scorer(mean_absolute_error)
    ).mean()
    return 1 - val


rf_bo = BayesianOptimization(
    rf_cv,
    {'gamma': (0.001, 10000),
     'reg_lambda': (0.001, 10000),
     'reg_alpha': (0.001, 10000),
     'max_depth': (5, 20),
     'subsample': (0.4, 1),
     'colsample_bytree': (0.4, 1)

     }
)

rf_bo.maximize()
# print result
# |   iter    |  target   | colsam... |   gamma   | max_depth | reg_alpha | reg_la... | subsample |
# -------------------------------------------------------------------------------------------------
# |  1        | -744.0    |  0.6154   |  3.265e+0 |  12.42    |  856.2    |  5.001e+0 |  0.7564   |
# |  2        | -683.1    |  0.6215   |  914.7    |  17.85    |  5.965e+0 |  2.227e+0 |  0.651    |
# |  3        | -778.5    |  0.5025   |  1.855e+0 |  15.93    |  27.13    |  7.946e+0 |  0.82     |
# |  4        | -885.1    |  0.8799   |  3.85e+03 |  5.627    |  3.283e+0 |  4.6e+03  |  0.6098   |
# |  5        | -722.9    |  0.4227   |  5.459e+0 |  12.78    |  1.947e+0 |  1.784e+0 |  0.6059   |
# |  6        | -829.4    |  0.9476   |  3.577e+0 |  8.52     |  2.728e+0 |  7.811e+0 |  0.6511   |
# |  7        | -712.7    |  0.9493   |  9.874e+0 |  11.57    |  7.815e+0 |  2.841e+0 |  0.7689   |
# |  8        | -797.3    |  0.5165   |  1.827e+0 |  13.81    |  4.203e+0 |  9.792e+0 |  0.8199   |
# |  9        | -773.7    |  0.6819   |  5.91e+03 |  7.567    |  3.823e+0 |  2.037e+0 |  0.5355   |
# |  10       | -703.6    |  0.9821   |  8.94e+03 |  19.61    |  6.121e+0 |  4.083e+0 |  0.8342   |
# |  11       | -686.6    |  0.6385   |  891.9    |  17.81    |  5.985e+0 |  2.209e+0 |  0.6096   |
# |  12       | -742.8    |  0.7259   |  633.2    |  12.56    |  2.079e+0 |  4.912e+0 |  0.7051   |
# |  13       | -755.3    |  0.7037   |  5.417e+0 |  7.31     |  1.765e+0 |  1.718e+0 |  0.9999   |
# |  14       | -791.7    |  0.5571   |  5.799e+0 |  11.85    |  3.637e+0 |  5.67e+03 |  0.5417   |
# |  15       | -747.6    |  0.949    |  2.665e+0 |  16.49    |  8.7e+03  |  5.306e+0 |  0.679    |
# |  16       | -764.7    |  0.5059   |  9.815e+0 |  11.2     |  7.812e+0 |  2.986e+0 |  0.4012   |
# |  17       | -666.7    |  0.8044   |  809.1    |  17.21    |  5.822e+0 |  2.222e+0 |  0.8272   |
# |  18       | -695.9    |  0.7116   |  1.004e+0 |  19.48    |  5.798e+0 |  2.19e+03 |  0.5437   |
# |  19       | -981.7    |  0.4193   |  803.3    |  5.702    |  4.094e+0 |  9.896e+0 |  0.421    |
# |  20       | -680.3    |  0.9379   |  1.071e+0 |  13.2     |  5.901e+0 |  2.183e+0 |  0.8881   |
# |  21       | -629.0    |  0.9348   |  1.293e+0 |  18.55    |  3.499e+0 |  673.7    |  0.6636   |
# |  22       | -887.2    |  0.698    |  5.231e+0 |  5.702    |  7.207e+0 |  7.635e+0 |  0.8603   |
# |  23       | -693.1    |  0.7637   |  861.9    |  15.85    |  5.94e+03 |  2.459e+0 |  0.6157   |
# |  24       | -684.6    |  0.6743   |  924.4    |  17.85    |  5.823e+0 |  2.035e+0 |  0.5516   |
# |  25       | -738.1    |  0.9917   |  793.9    |  9.869    |  5.854e+0 |  2.574e+0 |  0.7508   |
# |  26       | -827.3    |  0.5981   |  2.883e+0 |  6.172    |  4.577e+0 |  4.775e+0 |  0.9584   |
# |  27       | -721.9    |  0.8943   |  916.0    |  10.64    |  5.865e+0 |  2.41e+03 |  0.7399   |
# |  28       | -722.3    |  0.7228   |  609.1    |  9.791    |  5.79e+03 |  2.151e+0 |  0.9697   |
# |  29       | -757.0    |  0.5643   |  1.909e+0 |  8.71     |  4.3e+03  |  2.638e+0 |  0.6205   |
# |  30       | -649.6    |  0.8162   |  1.365e+0 |  13.05    |  3.524e+0 |  634.7    |  0.9489   |
# =================================================================================================


# run optimized model again
xgr_opt_BO = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, subsample=0.948,
                              colsample_bytree=0.81, max_depth=13, reg_alpha=3.5, reg_lambda=634,
                              gamma=1.3)
nfoldcross(5, xgr_opt_BO, x_train_all, y_train_all)


# print result
# Train mae: 591.6672297433674
# Val mae 662.8300053272558

# BayesianOptimization not works very well, at least in this case


# 4.4 plot learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_size=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('Training example')
    plt.ylabel('score')
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                                                            train_sizes=train_size,
                                                            scoring=make_scorer(mean_absolute_error))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()  # 区域
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r',
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    return plt


plot_learning_curve(xgr_opt, 'XGB model learning curve', x_train_all, y_train_all, cv=3,
                    n_jobs=1)
