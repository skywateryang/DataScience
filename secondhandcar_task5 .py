# -*- coding: utf-8 -*-

# Tianchi secondhand car saleprice prediction
# -  Fifth part - Model blending

# 5.1 import package and data
# --------------------------------------------------------------------------------------#
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV

data = pd.read_csv('data.csv')
y_train_all = data['price']
x_train_all = data.drop(['price'], axis=1)

test_all=pd.read_csv('test.csv')
test_label = test_all['SaleID']
test_data = test_all.drop(['SaleID','label'], axis=1)

# 5.2 build base models
# --------------------------------------------------------------------------------------#
X_train, X_test, y_train, y_test = train_test_split(x_train_all, y_train_all, test_size=0.3)
# model 1
xgr = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, subsample=0.8,
                       colsample_bytree=0.8, max_depth=10, reg_alpha=100, reg_lambda=0,
                       gamma=10000)
xgr.fit(X_train, y_train)
pre_xgb = xgr.predict(X_test)
MAE_xgb = mean_absolute_error(y_test, pre_xgb)
# MAE of model1 : 563

# model 2
lgr = lgb.LGBMRegressor(num_leaves=180, n_estimators=100, learning_rate=0.1, max_depth=12,
                        colsample_bytree=0.8, subsample=0.7)
lgr.fit(X_train, y_train)
pre_lgb = lgr.predict(X_test)
MAE_lgb = mean_absolute_error(y_test, pre_lgb)
# MAE of model2 : 582


# 5.3 model blending
# --------------------------------------------------------------------------------------#

# 5.3.1 weighted blending
# calculating weight
pre_weighted = (1 - MAE_lgb / (MAE_xgb + MAE_lgb)) * pre_lgb + (1 - MAE_xgb / (MAE_xgb + MAE_lgb)) * pre_xgb
pre_weighted[pre_weighted < 0] = 10
print('MAE of Weighted ensemble:', mean_absolute_error(y_test, pre_weighted))
# MAE of weighted blending : 554

# 5.3.2 model stacking
from sklearn.ensemble import StackingRegressor
estimators = [('xgb', xgr), ('lgb', lgr)]
clf_stacking = StackingRegressor(estimators=estimators, final_estimator=RidgeCV())
clf_stacking.fit(X_train, y_train)
pre_stacking = clf_stacking.predict(X_test)
pre_stacking[pre_stacking < 0] = 10
MAE_stacking = mean_absolute_error(y_test, pre_stacking)
print(MAE_stacking)

# MAE of model stacking: 558

# 5.4 output result
clf_stacking.fit(x_train_all, y_train_all)
predict = clf_stacking.predict(test_data)
predict[predict < 0] = 10
result = pd.DataFrame({'SaleID':test_label,'price':predict})
result.to_csv('result.csv',index=False)
