# -*- coding: utf-8 -*-

# Tianchi secondhand car saleprice prediction
# -  Third part - feature engineering


# 3.1 import package
# --------------------------------------------------------------------------------------#
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

pd.set_option('display.max_columns', 50)

# 3.2 import train and test data
# --------------------------------------------------------------------------------------#
train = pd.read_csv(r'used_car_train_20200313.csv', sep=' ')
test = pd.read_csv(r'used_car_testA_20200313.csv', sep=' ')

# 3.3 Dealing with abnormal values
# --------------------------------------------------------------------------------------#
train.loc[train.notRepairedDamage == '-', 'notRepairedDamage'] = 0.0
test.loc[test.notRepairedDamage == '-', 'notRepairedDamage'] = 0.0


# 3.4 Dealing with outliers
# --------------------------------------------------------------------------------------#
# define a new function to cap value beyond 3 sigma, which would be called in model building
def outlier(dataset, colname):
    ser = dataset[colname]
    mean = ser.mean()
    std = ser.std()
    upperlimit = mean + 3 * std
    lowerlimit = mean - 3 * std
    ser.loc[ser > upperlimit] = upperlimit
    ser.loc[ser < lowerlimit] = lowerlimit
    return ser


# 3.5 Dealing with missing value
# --------------------------------------------------------------------------------------#
# replace with most frequent value
train.loc[train.bodyType.isnull(), 'bodyType'] = 0.0
train.loc[train.fuelType.isnull(), 'fuelType'] = 0.0
train.loc[train.gearbox.isnull(), 'gearbox'] = 0.0
test.loc[test.bodyType.isnull(), 'bodyType'] = 0.0
test.loc[test.fuelType.isnull(), 'fuelType'] = 0.0
test.loc[test.gearbox.isnull(), 'gearbox'] = 0.0

# 3.6 feature construction
# --------------------------------------------------------------------------------------#
# extract city info
train['city'] = train['regionCode'].apply(lambda x: str(x)[:-3])
test['city'] = train['regionCode'].apply(lambda x: str(x)[:-3])

# build the feature of used days by substracting createdate by regdate
train['useddays'] = (pd.to_datetime(train.creatDate, format='%Y%m%d', errors='coerce') - pd.to_datetime(train.regDate,
                                                                                                        format='%Y%m%d',
                                                                                                        errors='coerce')).dt.days
test['useddays'] = (pd.to_datetime(test.creatDate, format='%Y%m%d', errors='coerce') - pd.to_datetime(test.regDate,
                                                                                                      format='%Y%m%d',
                                                                                                      errors='coerce')).dt.days
mean_useddays = (test.useddays.sum() + train.useddays.sum()) / (len(train) + len(test))
train.loc[train.useddays.isnull(), 'useddays'] = mean_useddays
test.loc[test.useddays.isnull(), 'useddays'] = mean_useddays

# build a set of feature using priori knowledge
train_gb = train.groupby("brand")
all_info = {}
for kind, kind_data in train_gb:
    info = {}
    info['brand_amount'] = len(kind_data)
    info['brand_price_max'] = kind_data.price.max()
    info['brand_price_median'] = kind_data.price.median()
    info['brand_price_min'] = kind_data.price.min()
    info['brand_price_sum'] = kind_data.price.sum()
    info['brand_price_std'] = kind_data.price.std()
    info['brand_price_average'] = round(kind_data.price.sum() / (len(kind_data) + 1), 2)
    all_info[kind] = info
brand_fe = pd.DataFrame(all_info).T.reset_index().rename(columns={"index": "brand"})
train = train.merge(brand_fe, how='left', on='brand')
test = test.merge(brand_fe, how='left', on='brand')

# 3.7 Remove useless columns
# --------------------------------------------------------------------------------------#
train.drop(['name', 'offerType', 'seller', 'regionCode', 'creatDate', 'regDate'], inplace=True, axis=1)
test.drop(['name', 'offerType', 'seller', 'regionCode', 'creatDate', 'regDate'], inplace=True, axis=1)

# 3.8 feature selection
# filter method
numeric_features = ['power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10',
                    'v_11', 'v_12', 'v_13', 'v_14', 'useddays', 'brand_amount', 'brand_price_max', 'brand_price_median',
                    'brand_price_min', 'brand_price_sum', 'brand_price_std', 'brand_price_average']
y_train = train['price']
x_train_num = train[numeric_features]
select = SelectPercentile(f_regression, percentile=70).fit(x_train_num, y_train)
x_train_selected = select.transform(x_train_num)
x_train_num.columns[select.get_support()]
print(*list(x_train_num.columns[select.get_support()]), sep='\n')

# auto selected feature by filter method
# power
# kilometer
# v_0
# v_3
# v_4
# v_5
# v_8
# v_9
# v_10
# v_11
# v_12
# useddays
# brand_price_max
# brand_price_median
# brand_price_sum
# brand_price_std
# brand_price_average


# wrapper method
# due to the high requirement on calculating ability, interrupt halfway
selector = RFE(estimator=LogisticRegression(), n_features_to_select=17).fit(x_train_num,y_train)
x_train_selected = select.transform(x_train_num)
x_train_num.columns[select.get_support()]
