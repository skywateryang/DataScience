# -*- coding: utf-8 -*-

# Tianchi secondhand car saleprice prediction
# -  Second part - EDA


# 2.1 import package
# --------------------------------------------------------------------------------------#
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
pd.set_option('display.max_columns',50)


# 2.2 import train and test data
# --------------------------------------------------------------------------------------#
train = pd.read_csv(r'used_car_train_20200313.csv', sep=' ')
test = pd.read_csv(r'used_car_testA_20200313.csv', sep=' ')

# view data shape and columns
print(train.columns)
print(train.shape)
print(train.head().append(train.tail()))

# name - 汽车编码
# regDate - 汽车注册时间
# model - 车型编码
# brand - 品牌
# bodyType - 车身类型
# fuelType - 燃油类型
# gearbox - 变速箱
# power - 汽车功率
# kilometer - 汽车行驶公里
# notRepairedDamage - 汽车有尚未修复的损坏
# regionCode - 看车地区编码
# seller - 销售方
# offerType - 报价类型
# creatDate - 广告发布时间
# price - 汽车价格
# 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13','v_14' 【匿名特征，包含v0-14在内15个匿名特征】


# 2.3 Overview the data structure
# --------------------------------------------------------------------------------------#
train.describe()
train.info()
test.describe()
test.info()

# train info
#      Column             Non-Null Count   Dtype
# ---  ------             --------------   -----
#  0   SaleID             150000 non-null  int64
#  1   name               150000 non-null  int64
#  2   regDate            150000 non-null  int64
#  3   model              149999 non-null  float64
#  4   brand              150000 non-null  int64
#  5   bodyType           145494 non-null  float64
#  6   fuelType           141320 non-null  float64
#  7   gearbox            144019 non-null  float64
#  8   power              150000 non-null  int64
#  9   kilometer          150000 non-null  float64
#  10  notRepairedDamage  150000 non-null  object
#  11  regionCode         150000 non-null  int64
#  12  seller             150000 non-null  int64
#  13  offerType          150000 non-null  int64
#  14  creatDate          150000 non-null  int64
#  15  price              150000 non-null  int64
#  16  v_0                150000 non-null  float64
#  17  v_1                150000 non-null  float64
#  18  v_2                150000 non-null  float64
#  19  v_3                150000 non-null  float64
#  20  v_4                150000 non-null  float64
#  21  v_5                150000 non-null  float64
#  22  v_6                150000 non-null  float64
#  23  v_7                150000 non-null  float64
#  24  v_8                150000 non-null  float64
#  25  v_9                150000 non-null  float64
#  26  v_10               150000 non-null  float64
#  27  v_11               150000 non-null  float64
#  28  v_12               150000 non-null  float64
#  29  v_13               150000 non-null  float64
#  30  v_14               150000 non-null  float64


# 2.4 Data missing and anomaly detection
# --------------------------------------------------------------------------------------#
train.isnull().sum().sort_values()
test.isnull().sum().sort_values()

# missing count for train
# model                   1
# bodyType             4506
# gearbox              5981
# fuelType             8680

# missing count for test
# bodyType             1413
# gearbox              1910
# fuelType             2893

# review category features one by one

# name, totally 99662 categories --> delete later
train.name.value_counts()

# regDate, totally 3894 categories
train.regDate.value_counts()

# totally 248 kind of model and 40 kind of brand
train.model.value_counts()
train.brand.value_counts()

# 8 kind of body, 7 kind of fuel and 2 kind of gear
train.bodyType.value_counts()
train.fuelType.value_counts()
train.gearbox.value_counts()

# 24324 Abnormal value '-' --> replace later
train.notRepairedDamage.value_counts()

# regioncode, 7905 categories --> delete later
train.regionCode.value_counts()

# seller, extremely skewed --> delete minor category later
train.seller.value_counts()

# offer type, all sample with the same value --> delete later
train.offerType.value_counts()

# create date, none of the samples share the same value --> delete later
train.creatDate.value_counts()


# 2.5 Study the distribution of response
# --------------------------------------------------------------------------------------#
# skewed distribution with long tail
train.price.hist(bins=100)

# well fit as normal dist after log
sns.distplot(np.log(train.price), kde=False, color="b",fit=st.norm)


# 2.6 analysis of float type features
# --------------------------------------------------------------------------------------#


