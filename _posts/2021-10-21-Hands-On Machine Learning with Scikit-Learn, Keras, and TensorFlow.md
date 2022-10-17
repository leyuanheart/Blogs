---
layout:     post
title:      Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow
subtitle:   Part 1
date:       2021-10-21
author:     Le Yuan
header-img: img/wolf.jpg
catalog: true
tags:
    - Machine Learning
    - Scikit Learn
    - TensorFlow
---

<img src="https://pic.imgdb.cn/item/6171505c2ab3f51d91c5b410.jpg" alt="1" style="zoom:40%;" />

开新坑，就是这本书了。这篇是介绍使用TensorFlow之前的所有内容。

## The Machine Learning Landscape

![1](https://pic.imgdb.cn/item/617150752ab3f51d91c5d824.png)

关于**“模型”**这个词，有时它指的是**“模型的类型”**，比如线性模型、多项式模型。有时是指**“特定的模型结构”**，比如有一个输入变量和一个输出变量的线性模型、多个输入一个输出的线性模型。训练模型是是指使用一种算法来找到模型的参数，使其能最好地拟合训练数据。

![2](https://pic.imgdb.cn/item/617150752ab3f51d91c5d82e.png)

![3](https://pic.imgdb.cn/item/617150752ab3f51d91c5d839.png)

The most important rule to remember is that the validation set and the test set must be as representative as possible of the data you expect to use in production。

In a famous 1996 paper, David Wolpert demonstrated that **if you make absolutely no assumption about the data, then there is no reason to prefer one model over any other.** This is called the **No Free Lunch (NFL)** theorem.

## End-to-End Machine Learning Project
Popular open data repositories

- [UC Irvine Machine Learning Repository](http://archive.ics.uci.edu/ml/)

- [Kaggle datasets](https://www.kaggle.com/datasets)

- [Amazon’s AWS datasets](https://registry.opendata.aws/)

Meta portals (they list open data repositories)

- [Data Portals](http://dataportals.org/)

- [OpenDataMonitor](http://opendatamonitor.eu/)

- [Quandl](http://quandl.com/)

Other pages listing many popular open data repositories 

- [Wikipedia’s list of Machine Learning datasets](https://homl.info/9)

- [Quora.com](https://homl.info/10)



- ![4](https://pic.imgdb.cn/item/617150752ab3f51d91c5d844.png)

The higher the norm index, the more it focuses on large values and neglects small ones.

<font color=red>We spent quite a bit of time on test set generation for a good reason: this is an often neglected but critical part of a Machine Learning project</font> (Random sampling or stratified sampling).



# Download the data


```python
import sys
sys.version
```




    '3.7.7 (default, May  6 2020, 11:45:54) [MSC v.1916 64 bit (AMD64)]'




```python
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from matplotlib.projections import Axes3D
```


```python
import os
import tarfile
import urllib.request


DOWNLOAD_ROOT = "https://github.com/ageron/handson-ml2/tree/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, 'housing.tgz')
    urllib.request.urlretrieve(housing_url, tgz_path)
    
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    
# fetch_housing_data()
```


```python
import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)

housing = load_housing_data()
```


```python
housing.head()
```

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-122.23</td>
      <td>37.88</td>
      <td>41.0</td>
      <td>880.0</td>
      <td>129.0</td>
      <td>322.0</td>
      <td>126.0</td>
      <td>8.3252</td>
      <td>452600.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-122.22</td>
      <td>37.86</td>
      <td>21.0</td>
      <td>7099.0</td>
      <td>1106.0</td>
      <td>2401.0</td>
      <td>1138.0</td>
      <td>8.3014</td>
      <td>358500.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-122.24</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1467.0</td>
      <td>190.0</td>
      <td>496.0</td>
      <td>177.0</td>
      <td>7.2574</td>
      <td>352100.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1274.0</td>
      <td>235.0</td>
      <td>558.0</td>
      <td>219.0</td>
      <td>5.6431</td>
      <td>341300.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1627.0</td>
      <td>280.0</td>
      <td>565.0</td>
      <td>259.0</td>
      <td>3.8462</td>
      <td>342200.0</td>
      <td>NEAR BAY</td>
    </tr>
  </tbody>
</table>
</div>


```python
housing.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20640 entries, 0 to 20639
    Data columns (total 10 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   longitude           20640 non-null  float64
     1   latitude            20640 non-null  float64
     2   housing_median_age  20640 non-null  float64
     3   total_rooms         20640 non-null  float64
     4   total_bedrooms      20433 non-null  float64
     5   population          20640 non-null  float64
     6   households          20640 non-null  float64
     7   median_income       20640 non-null  float64
     8   median_house_value  20640 non-null  float64
     9   ocean_proximity     20640 non-null  object 
    dtypes: float64(9), object(1)
    memory usage: 1.6+ MB



```python
housing.describe()
```

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20433.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-119.569704</td>
      <td>35.631861</td>
      <td>28.639486</td>
      <td>2635.763081</td>
      <td>537.870553</td>
      <td>1425.476744</td>
      <td>499.539680</td>
      <td>3.870671</td>
      <td>206855.816909</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.003532</td>
      <td>2.135952</td>
      <td>12.585558</td>
      <td>2181.615252</td>
      <td>421.385070</td>
      <td>1132.462122</td>
      <td>382.329753</td>
      <td>1.899822</td>
      <td>115395.615874</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-124.350000</td>
      <td>32.540000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>0.499900</td>
      <td>14999.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-121.800000</td>
      <td>33.930000</td>
      <td>18.000000</td>
      <td>1447.750000</td>
      <td>296.000000</td>
      <td>787.000000</td>
      <td>280.000000</td>
      <td>2.563400</td>
      <td>119600.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-118.490000</td>
      <td>34.260000</td>
      <td>29.000000</td>
      <td>2127.000000</td>
      <td>435.000000</td>
      <td>1166.000000</td>
      <td>409.000000</td>
      <td>3.534800</td>
      <td>179700.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>-118.010000</td>
      <td>37.710000</td>
      <td>37.000000</td>
      <td>3148.000000</td>
      <td>647.000000</td>
      <td>1725.000000</td>
      <td>605.000000</td>
      <td>4.743250</td>
      <td>264725.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>-114.310000</td>
      <td>41.950000</td>
      <td>52.000000</td>
      <td>39320.000000</td>
      <td>6445.000000</td>
      <td>35682.000000</td>
      <td>6082.000000</td>
      <td>15.000100</td>
      <td>500001.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
housing['ocean_proximity'].value_counts()
```




    <1H OCEAN     9136
    INLAND        6551
    NEAR OCEAN    2658
    NEAR BAY      2290
    ISLAND           5
    Name: ocean_proximity, dtype: int64




```python
%matplotlib inline
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20, 15))
plt.show()
```


![png](https://pic.imgdb.cn/item/61715cfe2ab3f51d91d5407c.png)



```python
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
```

# stratified sampling


```python
import numpy as np
housing['income_cat'] = pd.cut(housing['median_income'], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
housing['income_cat'].hist()
```




    <AxesSubplot:>




![png](https://pic.imgdb.cn/item/61715cfe2ab3f51d91d54081.png)



```python
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
```


```python
for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)
```

# Discover and visualize the data to gain insights


```python
housing = strat_train_set.copy()
```


```python
housing.plot(kind='scatter', x='longitude', y='latitude')
```




    <AxesSubplot:xlabel='longitude', ylabel='latitude'>




![png](https://pic.imgdb.cn/item/61715cfe2ab3f51d91d54089.png)



```python
corr_matrix = housing.corr()
```


```python
from pandas.plotting import scatter_matrix
attributes=["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes],figsize=(12, 8))
```




    array([[<AxesSubplot:xlabel='median_house_value', ylabel='median_house_value'>,
            <AxesSubplot:xlabel='median_income', ylabel='median_house_value'>,
            <AxesSubplot:xlabel='total_rooms', ylabel='median_house_value'>,
            <AxesSubplot:xlabel='housing_median_age', ylabel='median_house_value'>],
           [<AxesSubplot:xlabel='median_house_value', ylabel='median_income'>,
            <AxesSubplot:xlabel='median_income', ylabel='median_income'>,
            <AxesSubplot:xlabel='total_rooms', ylabel='median_income'>,
            <AxesSubplot:xlabel='housing_median_age', ylabel='median_income'>],
           [<AxesSubplot:xlabel='median_house_value', ylabel='total_rooms'>,
            <AxesSubplot:xlabel='median_income', ylabel='total_rooms'>,
            <AxesSubplot:xlabel='total_rooms', ylabel='total_rooms'>,
            <AxesSubplot:xlabel='housing_median_age', ylabel='total_rooms'>],
           [<AxesSubplot:xlabel='median_house_value', ylabel='housing_median_age'>,
            <AxesSubplot:xlabel='median_income', ylabel='housing_median_age'>,
            <AxesSubplot:xlabel='total_rooms', ylabel='housing_median_age'>,
            <AxesSubplot:xlabel='housing_median_age', ylabel='housing_median_age'>]],
          dtype=object)




![png](https://pic.imgdb.cn/item/61715cfe2ab3f51d91d54097.png)


# Data Cleaning


```python
housing=strat_train_set.drop("median_house_value",axis=1)   # note that drop create a copy of the data
housing_labels=strat_train_set["median_house_value"].copy()
```


```python
housing.info() 
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 16512 entries, 17606 to 15775
    Data columns (total 9 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   longitude           16512 non-null  float64
     1   latitude            16512 non-null  float64
     2   housing_median_age  16512 non-null  float64
     3   total_rooms         16512 non-null  float64
     4   total_bedrooms      16354 non-null  float64
     5   population          16512 non-null  float64
     6   households          16512 non-null  float64
     7   median_income       16512 non-null  float64
     8   ocean_proximity     16512 non-null  object 
    dtypes: float64(8), object(1)
    memory usage: 1.3+ MB


total_bedrooms has some missing values. Three options:
1. Get rid of the corresponding districts.
2. Get rid of the whole attribute.
3. Set the values to some value (zero, the mean, the median, etc.).


```python
housing.dropna(subset=['total_bedrooms'])  # option 1
housing.drop('total_bedrooms', axis=1)     # option 2

median = housing['total_bedrooms'].median() # option 3
housing['total_bedrooms'].fillna(median, inplace=True)
```

If you choose option 3, you should compute the median value on the training set and use it to fill the missing values in the training set. Don’t forget to save the median value that you have computed. <font color=red>You will need it later to replace missing values in the test set when you want to evaluate your system</font>, and also once the system goes live to replace missing values in new data.


```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
housing_num = housing.drop('ocean_proximity', axis=1) # create a copy of the data without the text attributes
imputer.fit(housing_num)
```




    SimpleImputer(strategy='median')




```python
print(imputer.statistics_)
print(housing_num.median().values)
```

    [-118.51     34.26     29.     2119.5     433.     1164.      408.
        3.5409]
    [-118.51     34.26     29.     2119.5     433.     1164.      408.
        3.5409]



```python
X = imputer.transform(housing_num)  
# The result is a plain NumPy array. You can transform it to pandas DataFrame
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
```

# Handle Text and Categorical Attributes


```python
housing_cat = housing[['ocean_proximity']]
housing_cat.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17606</th>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>18632</th>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>14650</th>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>3230</th>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>3555</th>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>19480</th>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>8879</th>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>13685</th>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>4937</th>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>4861</th>
      <td>&lt;1H OCEAN</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]
```




    array([[0.],
           [0.],
           [4.],
           [1.],
           [0.],
           [1.],
           [0.],
           [1.],
           [0.],
           [0.]])




```python
ordinal_encoder.categories_
```




    [array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
           dtype=object)]




```python
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot   # call housing_cat_1hot.toarray() to get a (dense) NumPy array
```




    <16512x5 sparse matrix of type '<class 'numpy.float64'>'
    	with 16512 stored elements in Compressed Sparse Row format>



# Feature Scaling

1. Min-Max scaling (Normalization): MinMaxScaler, feature_range hyperparameter to change  the range
2. Standardization                : StandardScaler

Standardization does not bound values to a specific range, which may be a problem for some algorithms(e.g., neural networks often expect an input value ranging from 0 to 1). However, standardization is much less affected by outliers.

# Transformation Pipelines


```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
                        ('imputer', SimpleImputer(strategy='median')),
                        ('std_scaler', StandardScaler())
                         ])
housing_num_tr = num_pipeline.fit_transform(housing_num)
```


```python
from sklearn.compose import ColumnTransformer  # Scikit-Learin version 0.20

num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', OneHotEncoder(), cat_attribs)
])

housing_prepared = full_pipeline.fit_transform(housing)
```

# Select and Train a Model


```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
```




    LinearRegression()




```python
 # let's try the full preprocessing pipeline on a few training instances
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))
```

    Predictions: [211574.39523833 321345.10513719 210947.519838    61921.01197837
     192362.32961119]
    Labels: [286600.0, 340600.0, 196900.0, 46300.0, 254500.0]



```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)   # Note: since Scikit-Learn 0.22, you can get the RMSE directly by calling the mean_squared_error() function with squared=False.
print(lin_rmse)
lin_mae = mean_absolute_error(housing_labels, housing_predictions)
print(lin_mae)
```

    69050.98178244587
    49906.94142223287


# Better Evaluation Using Cross-Validation


```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10) # 10-fold cross validation
'''
Scikit-Learn’s cross-validation features expect a utility function (greater is better)
rather than a cost function (lower is better)
'''
lin_rmse_scores = np.sqrt(-scores)
lin_rmse_scores
```




    array([67450.42057782, 67329.50264436, 68361.84864912, 74639.88837894,
           68314.56738182, 71628.61410355, 65361.14176205, 68571.62738037,
           72476.18028894, 68098.06828865])



Notice that cross-validation allows you to get not only an estimate of the performance of your model, but also a measure of how precise this estimate is (i.e., its standard deviation).You would not have this information if you just used one validation set. But crossvalidation comes at the cost of training the model several times, so it is not always possible.

You can easily save Scikit-Learn models by using Python’s pickle module or by using the joblib library, which is more efficient at serializing large NumPy arrays.


```python
import joblib
joblib.dump(lin_reg, 'lin.pkl')

# add later...
lin_reg = joblib.load('lin.pkl')
```




    ['lin.pkl']



# Fine-Tune Your Model

## Grid Search


```python
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import GridSearchCV
param_grid = [{'n_estimators':[3, 10, 30], 'max_features':[2, 4, 6, 8]},
              {'bootstrap':[False], 'n_estimators':[3, 10], 'max_features':[2, 3, 4]}]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)

grid_search.fit(housing_prepared,housing_labels)
```


```python
grid_search.best_params_
```


```python
grid_search.best_estimator_
```


```python
cvres=grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
```

If GridSearchCV is initialized with refit=True (which is the default), then once it finds the best estimator using cross-validation, it retrains it on the whole training set.

# Evaluate Your System on the Test Set

run your full_pipeline to transform the data **(call transform(), not fit_transform()—you do not want to fit the test set!)**, and evaluate the final model on the test set:


```python
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

final_rmse
```

## Confidence Interval


```python
from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test)**2

np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1, loc=squared_errors.mean(), scale=stats.sem(squared_errors)))
```

# Classification


```python
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.keys()
```




    dict_keys(['data', 'target', 'frame', 'categories', 'feature_names', 'target_names', 'DESCR', 'details', 'url'])




```python
X, y = mnist["data"], mnist["target"]
X.shape, y.shape
```




    ((70000, 784), (70000,))




```python
import numpy as np
y = y.astype(np.uint8)
```


```python
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
```

## Train a binary classifier


```python
y_train_5 = (y_train==5)
y_test_5 = (y_test==5)
```


```python
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)   # linear model
sgd_clf.fit(X_train, y_train_5)
```




    SGDClassifier(random_state=42)




```python
sgd_clf.predict(X_train[0, np.newaxis])
```




    array([ True])




```python
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')
```




    array([0.95035, 0.96035, 0.9604 ])



The following code does roughly the same thing as `cross_val_score()` function


```python
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))
```

    0.9669
    0.91625
    0.96785



```python
from sklearn.base import BaseEstimator
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")
```




    array([0.91125, 0.90855, 0.90915])



# Confusion Matrix


```python
from sklearn.model_selection import cross_val_predict

y_trained_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
```


```python
from sklearn.metrics import confusion_matrix

confusion_matrix(y_train_5, y_trained_pred)
```




    array([[53892,   687],
           [ 1891,  3530]], dtype=int64)



![5](https://pic.imgdb.cn/item/617150752ab3f51d91c5d84e.png)


```python
from sklearn.metrics import precision_score, recall_score

print(precision_score(y_train_5, y_trained_pred))
print(recall_score(y_train_5, y_trained_pred))
```

    0.8370879772350012
    0.6511713705958311


![6](https://pic.imgdb.cn/item/617155b82ab3f51d91cc1590.png)

The harmonic mean gives much more weights to low values and favors classifiers that have similar precision and recall.


```python
from sklearn.metrics import f1_score

f1_score(y_train_5, y_trained_pred)
```




    0.7325171197343846



## Precsion/Recall Trade-off


```python
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method='decision_function')
```


```python
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
```


```python
precisions.shape, thresholds.shape
```




    ((59967,), (59966,))




```python
import matplotlib.pyplot as plt
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend(loc="center right", fontsize=16) # Not shown in the book
    plt.xlabel("Threshold", fontsize=16)        # Not shown
    plt.grid(True)                              # Not shown
    plt.axis([-50000, 50000, 0, 1])             # Not shown
```


```python
recall_90_precision = recalls[np.argmax(precisions >= 0.90)]
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]
```


```python
recall_90_precision, threshold_90_precision
```




    (0.4799852425751706, 3370.0194991439557)




```python
plt.figure(figsize=(8, 4))                                                                  
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

plt.plot([threshold_90_precision, threshold_90_precision], [0., 0.9], "r:")                 
plt.plot([-50000, threshold_90_precision], [0.9, 0.9], "r:")                                
plt.plot([-50000, threshold_90_precision], [recall_90_precision, recall_90_precision], "r:")
plt.plot([threshold_90_precision], [0.9], "ro")                                             
plt.plot([threshold_90_precision], [recall_90_precision], "ro")   
```




    [<matplotlib.lines.Line2D at 0x23e02be8c48>]




![png](https://pic.imgdb.cn/item/61715cfe2ab3f51d91d540a1.png)



```python
def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)

plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)
plt.plot([recall_90_precision, recall_90_precision], [0., 0.9], "r:")
plt.plot([0.0, recall_90_precision], [0.9, 0.9], "r:")
plt.plot([recall_90_precision], [0.9], "ro")
```




    [<matplotlib.lines.Line2D at 0x23e03c3d908>]




![png](https://pic.imgdb.cn/item/61715d682ab3f51d91d5a396.png)


## The ROC Curve

![7](https://pic.imgdb.cn/item/617155b82ab3f51d91cc1595.png)


```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
```


```python
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16)
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)
    plt.grid(True)

plt.figure(figsize=(8, 6))
plot_roc_curve(fpr, tpr)
fpr_90 = fpr[np.argmax(tpr >= recall_90_precision)]
plt.plot([fpr_90, fpr_90], [0., recall_90_precision], "r:")
plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:")
plt.plot([fpr_90], [recall_90_precision], "ro")
```




    [<matplotlib.lines.Line2D at 0x23e03cc7e48>]




![png](https://pic.imgdb.cn/item/61715d682ab3f51d91d5a39b.png)



```python
from sklearn.metrics import roc_auc_score

roc_auc_score(y_train_5, y_scores)
```




    0.9604938554008616



![8](https://pic.imgdb.cn/item/617155b82ab3f51d91cc15ad.png)


```python
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method="predict_proba")
```


```python
y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)
```


```python
recall_for_forest = tpr_forest[np.argmax(fpr_forest >= fpr_90)]

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.plot([fpr_90, fpr_90], [0., recall_90_precision], "r:")
plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:")
plt.plot([fpr_90], [recall_90_precision], "ro")
plt.plot([fpr_90, fpr_90], [0., recall_for_forest], "r:")
plt.plot([fpr_90], [recall_for_forest], "ro")
plt.grid(True)
plt.legend(loc="lower right", fontsize=16)
```




    <matplotlib.legend.Legend at 0x23e04feb2c8>




![png](https://pic.imgdb.cn/item/61715d682ab3f51d91d5a3a4.png)


# Multiclass Claasification

![9](https://pic.imgdb.cn/item/617155b82ab3f51d91cc15b5.png)


```python
from sklearn.svm import SVC

svm_clf = SVC(gamma="auto", random_state=42)
svm_clf.fit(X_train[:1000], y_train[:1000])     # y_train, not y_train_5. Under the hood scikit-learn uses OvO strategy
svm_clf.predict([X_train[0]])
```




    array([5], dtype=uint8)




```python
some_digit_scores = svm_clf.decision_function([X_train[0]])
some_digit_scores
```




    array([[ 2.81585438,  7.09167958,  3.82972099,  0.79365551,  5.8885703 ,
             9.29718395,  1.79862509,  8.10392157, -0.228207  ,  4.83753243]])




```python
np.argmax(some_digit_scores)
```




    5




```python
svm_clf.classes_
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8)




```python
svm_clf.classes_[5]
```




    5



![10](https://pic.imgdb.cn/item/617155b82ab3f51d91cc15bf.png)

If you want to force Scikit-Learn to use one-versus-one or one-versus-the-rest, you can use the `OneVsOneClassifier` or `OneVsRestClassifier` classes.


```python
from sklearn.multiclass import OneVsRestClassifier
ovr_clf = OneVsRestClassifier(SVC(gamma="auto", random_state=42))
ovr_clf.fit(X_train[:1000], y_train[:1000])
```




    OneVsRestClassifier(estimator=SVC(gamma='auto', random_state=42))




```python
len(ovr_clf.estimators_)
```




    10




```python
sgd_clf.fit(X_train, y_train)
sgd_clf.predict([X_train[0]])

# This time Scikit-Learn did not have to run OvR or OvO because SGD classifiers can directly classify instances into multiple classes.
```




    array([3], dtype=uint8)




```python
# Simply scaling the inputs increases accuracy
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
```

# Error Analysis


```python
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx
```




    array([[5577,    0,   22,    5,    8,   43,   36,    6,  225,    1],
           [   0, 6400,   37,   24,    4,   44,    4,    7,  212,   10],
           [  27,   27, 5220,   92,   73,   27,   67,   36,  378,   11],
           [  22,   17,  117, 5227,    2,  203,   27,   40,  403,   73],
           [  12,   14,   41,    9, 5182,   12,   34,   27,  347,  164],
           [  27,   15,   30,  168,   53, 4444,   75,   14,  535,   60],
           [  30,   15,   42,    3,   44,   97, 5552,    3,  131,    1],
           [  21,   10,   51,   30,   49,   12,    3, 5684,  195,  210],
           [  17,   63,   48,   86,    3,  126,   25,   10, 5429,   44],
           [  25,   18,   30,   64,  118,   36,    1,  179,  371, 5107]],
          dtype=int64)




```python
plt.matshow(conf_mx, cmap=plt.cm.gray)
```




    <matplotlib.image.AxesImage at 0x23e0508d108>




![png](https://pic.imgdb.cn/item/61715d682ab3f51d91d5a3af.png)



```python
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
```


```python
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
```




    <matplotlib.image.AxesImage at 0x23e050f8088>




![png](https://pic.imgdb.cn/item/61715d682ab3f51d91d5a3bc.png)



```python
# EXTRA
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = plt.cm.binary, **options)
    plt.axis("off")
```


```python
cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

plt.figure(figsize=(8,8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
```


![png](https://pic.imgdb.cn/item/61715da92ab3f51d91d5e1d2.png)


# Multilabel Classification


```python
from sklearn.neighbors import  KNeighborsClassifier

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
```




    KNeighborsClassifier()




```python
knn_clf.predict([X_train[0]])
```




    array([[False,  True]])



![11](https://pic.imgdb.cn/item/617156312ab3f51d91ccb324.png)


```python
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average="macro")   # average='weighted": weight by the number of instances with that target label
```




    0.976410265560605



# Multioutput-Multiclass Classification

each label is multiclass

![12](https://pic.imgdb.cn/item/617156312ab3f51d91ccb32a.png)


```python
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise

noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise

y_train_mod = X_train
y_test_mod = X_test
```


```python
plt.subplot(121); plot_digits([X_test_mod[0]])
plt.subplot(122); plot_digits([y_test_mod[0]])
```


![png](https://pic.imgdb.cn/item/61715da92ab3f51d91d5e1da.png)



```python
knn_clf.fit(X_train_mod, y_train_mod)

clean_digit = knn_clf.predict([X_test_mod[0]])

plot_digits(clean_digit)
```


![png](https://pic.imgdb.cn/item/61715da92ab3f51d91d5e1e0.png)



# Training Models




```python
import numpy as np
import matplotlib.pyplot as plt

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print(theta_best)

X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance
y_predict = X_new_b.dot(theta_best)

plt.plot(X, y, "b.")
plt.plot(X_new, y_predict, "r-")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15])
```

    [[4.09039014]
     [2.94695794]]





    (0.0, 2.0, 0.0, 15.0)




![png](https://pic.imgdb.cn/item/61715da92ab3f51d91d5e1e7.png)



```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_
```




    (array([3.89997096]), array([[3.11092901]]))



The `LinearRegression` class is based on the `scipy.linalg.lstsq()` function (the name stands for "least squares"), which you could call directly:


```python
theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
theta_best_svd
```




    array([[3.89997096],
           [3.11092901]])



This function computes $\mathbf{X}^+\mathbf{y}$, where $\mathbf{X}^{+}$ is the pseudoinverse of $\mathbf{X}$ (specifically the Moore-Penrose inverse). You can use `np.linalg.pinv()` to compute the pseudoinverse directly:


```python
np.linalg.pinv(X_b).dot(y)
```




    array([[3.89997096],
           [3.11092901]])



![13](https://pic.imgdb.cn/item/617156312ab3f51d91ccb330.png)

![14](https://pic.imgdb.cn/item/617156312ab3f51d91ccb339.png)

# Gradient Descent

![15](https://pic.imgdb.cn/item/617156312ab3f51d91ccb342.png)

![16](https://pic.imgdb.cn/item/617156612ab3f51d91ccf1b1.png)


```python
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())

sgd_reg.intercept_, sgd_reg.coef_
```




    (array([4.08705855]), array([3.01469303]))



# Polinomial Regression


```python
import numpy as np
import numpy.random as rnd

np.random.seed(42)

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
```


```python
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
print(X[0])
print(X_poly[0])
```

    [-0.75275929]
    [-0.75275929  0.56664654]


![17](https://pic.imgdb.cn/item/617156612ab3f51d91ccf1ba.png)

# Learning Curve


```python
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Training set size", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)
```


```python
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)
plt.axis([0, 80, 0, 3])
```




    (0.0, 80.0, 0.0, 3.0)




![png](https://pic.imgdb.cn/item/61715da92ab3f51d91d5e1f1.png)


These learning curves are typical of a model that’s underfitting. Both curves have reached a plateau; they are close and fairly high.

If your model is underfitting the training data, adding more training examples will not help. You need to use a more complex model or come up with better features.


```python
from sklearn.pipeline import Pipeline

polynomial_regression = Pipeline([
        ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
        ("lin_reg", LinearRegression()),
    ])

plot_learning_curves(polynomial_regression, X, y)
plt.axis([0, 80, 0, 3])   
```




    (0.0, 80.0, 0.0, 3.0)




![png](https://pic.imgdb.cn/item/61715de62ab3f51d91d61765.png)


![18](https://pic.imgdb.cn/item/617156612ab3f51d91ccf1c3.png)

# Regularized Linear Models

![19](https://pic.imgdb.cn/item/617156612ab3f51d91ccf1ce.png)


```python
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
m = 20
X = 3 * np.random.rand(m, 1)
y = 1 + 0.5 * X + np.random.randn(m, 1) / 1.5
X_new = np.linspace(0, 3, 100).reshape(100, 1)
```

## Ridge Regression


```python
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=1, solver="cholesky", random_state=42)
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])
```




    array([[1.55071465]])




```python
# using stochastic graident descent
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(penalty="l2", max_iter=1000, tol=1e-3, random_state=42)
sgd_reg.fit(X, y.ravel())
sgd_reg.predict([[1.5]])
```




    array([1.47012588])




```python
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

def plot_model(model_class, polynomial, alphas, **model_kargs):
    for alpha, style in zip(alphas, ("b-", "g--", "r:")):
        model = model_class(alpha, **model_kargs) if alpha > 0 else LinearRegression()
        if polynomial:
            model = Pipeline([
                    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
                    ("std_scaler", StandardScaler()),
                    ("regul_reg", model),
                ])
        model.fit(X, y)
        y_new_regul = model.predict(X_new)
        lw = 2 if alpha > 0 else 1
        plt.plot(X_new, y_new_regul, style, linewidth=lw, label=r"$\alpha = {}$".format(alpha))
    plt.plot(X, y, "b.", linewidth=3)
    plt.legend(loc="upper left", fontsize=15)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 3, 0, 4])

plt.figure(figsize=(8,4))
plt.subplot(121)
plot_model(Ridge, polynomial=False, alphas=(0, 10, 100), random_state=42)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.subplot(122)
plot_model(Ridge, polynomial=True, alphas=(0, 10**-5, 1), random_state=42)
```


![png](https://pic.imgdb.cn/item/61715de62ab3f51d91d6176c.png)


## Lasso


```python
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
lasso_reg.predict([[1.5]])
```




    array([1.53788174])



## elastic Net

![20](https://pic.imgdb.cn/item/617156612ab3f51d91ccf1d7.png)


```python
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
elastic_net.fit(X, y)
elastic_net.predict([[1.5]])
```




    array([1.54333232])



# Early Stopping

![21](https://pic.imgdb.cn/item/6171569e2ab3f51d91cd37ff.png)


```python
from sklearn.model_selection import train_test_split
np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 2 + X + 0.5 * X**2 + np.random.randn(m, 1)

X_train, X_val, y_train, y_val = train_test_split(X[:50], y[:50].ravel(), test_size=0.5, random_state=10)
```


```python
from sklearn.metrics import mean_squared_error
from copy import deepcopy  # from sklearn.base import clone

poly_scaler = Pipeline([
    ('poly_features', PolynomialFeatures(degree=90, include_bias=False)),
    ('std_scaler', StandardScaler())
])

X_train_poly_scaled = poly_scaler.fit_transform(X_train)
X_val_poly_scaled = poly_scaler.fit_transform(X_val)

sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True, penalty=None, learning_rate='constant', eta0=5e-4, random_state=42)

minimum_val_error = float('inf')
best_epoch = None
best_model = None

for epoch in range(1000):
    sgd_reg.fit(X_train_poly_scaled, y_train) # continues where it left off beacause of the warm_start being True
    y_val_predict = sgd_reg.predict(X_val_poly_scaled)
    val_error = mean_squared_error(y_val, y_val_predict)
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = deepcopy(sgd_reg)
```


```python
sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True,
                       penalty=None, learning_rate="constant", eta0=0.0005, random_state=42)

n_epochs = 1000
train_errors, val_errors = [], []
for epoch in range(n_epochs):
    sgd_reg.fit(X_train_poly_scaled, y_train)
    y_train_predict = sgd_reg.predict(X_train_poly_scaled)
    y_val_predict = sgd_reg.predict(X_val_poly_scaled)
    train_errors.append(mean_squared_error(y_train, y_train_predict))
    val_errors.append(mean_squared_error(y_val, y_val_predict))

best_epoch = np.argmin(val_errors)
best_val_rmse = np.sqrt(val_errors[best_epoch])

plt.annotate('Best model',
             xy=(best_epoch, best_val_rmse),
             xytext=(best_epoch, best_val_rmse + 1),
             ha="center",
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=16,
            )

best_val_rmse -= 0.03  # just to make the graph look better
plt.plot([0, n_epochs], [best_val_rmse, best_val_rmse], "k:", linewidth=2)
plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Validation set")
plt.plot(np.sqrt(train_errors), "r--", linewidth=2, label="Training set")
plt.legend(loc="upper right", fontsize=14)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("RMSE", fontsize=14)
```




    Text(0, 0.5, 'RMSE')




![png](https://pic.imgdb.cn/item/61715de62ab3f51d91d61772.png)


# Lasso versus Ridge regularizaiton


```python
t1a, t1b, t2a, t2b = -1, 3, -1.5, 1.5

t1s = np.linspace(t1a, t1b, 500)
t2s = np.linspace(t2a, t2b, 500)
t1, t2 = np.meshgrid(t1s, t2s)
T = np.c_[t1.ravel(), t2.ravel()]
Xr = np.array([[1, 1], [1, -1], [1, 0.5]])
yr = 2 * Xr[:, :1] + 0.5 * Xr[:, 1:]

J = (1/len(Xr) * np.sum((T.dot(Xr.T) - yr.T)**2, axis=1)).reshape(t1.shape)  # 每一对(theta1, theta2)对应的MSE

N1 = np.linalg.norm(T, ord=1, axis=1).reshape(t1.shape)
N2 = np.linalg.norm(T, ord=2, axis=1).reshape(t1.shape)

t_min_idx = np.unravel_index(np.argmin(J), J.shape)   
t1_min, t2_min = t1[t_min_idx], t2[t_min_idx]

t_init = np.array([[0.25], [-1]])
```


```python
def bgd_path(theta, X, y, l1, l2, core = 1, eta = 0.05, n_iterations = 200):
    path = [theta]
    for iteration in range(n_iterations):
        gradients = core * 2/len(X) * X.T.dot(X.dot(theta) - y) + l1 * np.sign(theta) + l2 * theta
        theta = theta - eta * gradients
        path.append(theta)
    return np.array(path)

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10.1, 8))
for i, N, l1, l2, title in ((0, N1, 2., 0, "Lasso"), (1, N2, 0,  2., "Ridge")):
    JR = J + l1 * N1 + l2 * 0.5 * N2**2
    
    tr_min_idx = np.unravel_index(np.argmin(JR), JR.shape)
    t1r_min, t2r_min = t1[tr_min_idx], t2[tr_min_idx]

    levelsJ=(np.exp(np.linspace(0, 1, 20)) - 1) * (np.max(J) - np.min(J)) + np.min(J)
    levelsJR=(np.exp(np.linspace(0, 1, 20)) - 1) * (np.max(JR) - np.min(JR)) + np.min(JR)
    levelsN=np.linspace(0, np.max(N), 10)
    
    path_J = bgd_path(t_init, Xr, yr, l1=0, l2=0)
    path_JR = bgd_path(t_init, Xr, yr, l1, l2)
    path_N = bgd_path(np.array([[2.0], [0.5]]), Xr, yr, np.sign(l1)/3, np.sign(l2), core=0)

    ax = axes[i, 0]
    ax.grid(True)
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    ax.contourf(t1, t2, N / 2., levels=levelsN)
    ax.plot(path_N[:, 0], path_N[:, 1], "y--")
    ax.plot(0, 0, "ys")
    ax.plot(t1_min, t2_min, "ys")
    ax.set_title(r"$\ell_{}$ penalty".format(i + 1), fontsize=16)
    ax.axis([t1a, t1b, t2a, t2b])
    if i == 1:
        ax.set_xlabel(r"$\theta_1$", fontsize=16)
    ax.set_ylabel(r"$\theta_2$", fontsize=16, rotation=0)

    ax = axes[i, 1]
    ax.grid(True)
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    ax.contourf(t1, t2, JR, levels=levelsJR, alpha=0.9)
    ax.plot(path_JR[:, 0], path_JR[:, 1], "w-o")
    ax.plot(path_N[:, 0], path_N[:, 1], "y--")
    ax.plot(0, 0, "ys")
    ax.plot(t1_min, t2_min, "ys")
    ax.plot(t1r_min, t2r_min, "rs")
    ax.set_title(title, fontsize=16)
    ax.axis([t1a, t1b, t2a, t2b])
    if i == 1:
        ax.set_xlabel(r"$\theta_1$", fontsize=16)

```


![png](https://pic.imgdb.cn/item/61715de62ab3f51d91d6177b.png)


# Logistic Regression


```python
from sklearn import datasets
iris = datasets.load_iris()
# list(iris.keys())
# print(iris.DESCR)
X = iris["data"][:, 3:]  # petal width
y = (iris["target"] == 2).astype(int)  # 1 if Iris virginica, else 0# 
```


```python
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(solver="lbfgs", random_state=42)
log_reg.fit(X, y)
```




    LogisticRegression(random_state=42)




```python
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
decision_boundary = X_new[y_proba[:, 1] >= 0.5][0][0]

plt.figure(figsize=(8, 3))
plt.plot(X[y==0], y[y==0], "bs")
plt.plot(X[y==1], y[y==1], "g^")
plt.plot([decision_boundary, decision_boundary], [-1, 2], "k:", linewidth=2)
plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris virginica")
plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris virginica")
plt.text(decision_boundary + 0.02, 0.15, "Decision  boundary", fontsize=14, color="k", ha="center")
plt.arrow(decision_boundary, 0.08, -0.3, 0, head_width=0.05, head_length=0.1, fc='b', ec='b')
plt.arrow(decision_boundary, 0.92, 0.3, 0, head_width=0.05, head_length=0.1, fc='g', ec='g')
plt.xlabel("Petal width (cm)", fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 3, -0.02, 1.02])
```




    (0.0, 3.0, -0.02, 1.02)




![png](https://pic.imgdb.cn/item/61715de62ab3f51d91d61787.png)



```python
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"] == 2).astype(int)

log_reg = LogisticRegression(solver="lbfgs", C=10**10, random_state=42)
log_reg.fit(X, y)

x0, x1 = np.meshgrid(
        np.linspace(2.9, 7, 500).reshape(-1, 1),
        np.linspace(0.8, 2.7, 200).reshape(-1, 1),
    )
X_new = np.c_[x0.ravel(), x1.ravel()]

y_proba = log_reg.predict_proba(X_new)

plt.figure(figsize=(10, 4))
plt.plot(X[y==0, 0], X[y==0, 1], "bs")
plt.plot(X[y==1, 0], X[y==1, 1], "g^")

zz = y_proba[:, 1].reshape(x0.shape)
contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)


left_right = np.array([2.9, 7])
boundary = -(log_reg.coef_[0][0] * left_right + log_reg.intercept_[0]) / log_reg.coef_[0][1]

plt.clabel(contour, inline=1, fontsize=12)
plt.plot(left_right, boundary, "k--", linewidth=3)
plt.text(3.5, 1.5, "Not Iris virginica", fontsize=14, color="b", ha="center")
plt.text(6.5, 2.3, "Iris virginica", fontsize=14, color="g", ha="center")
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.axis([2.9, 7, 0.8, 2.7])
```




    (2.9, 7.0, 0.8, 2.7)




![png](https://pic.imgdb.cn/item/61715e252ab3f51d91d64ba9.png)


![22](https://pic.imgdb.cn/item/6171569e2ab3f51d91cd3808.png)

# Softmax Regression

![23](https://pic.imgdb.cn/item/6171569e2ab3f51d91cd380e.png)


```python
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = iris["target"]

softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10, random_state=42)
softmax_reg.fit(X, y)
```




    LogisticRegression(C=10, multi_class='multinomial', random_state=42)




```python
x0, x1 = np.meshgrid(
        np.linspace(0, 8, 500).reshape(-1, 1),
        np.linspace(0, 3.5, 200).reshape(-1, 1),
    )
X_new = np.c_[x0.ravel(), x1.ravel()]


y_proba = softmax_reg.predict_proba(X_new)
y_predict = softmax_reg.predict(X_new)

zz1 = y_proba[:, 1].reshape(x0.shape)
zz = y_predict.reshape(x0.shape)

plt.figure(figsize=(10, 4))
plt.plot(X[y==2, 0], X[y==2, 1], "g^", label="Iris virginica")
plt.plot(X[y==1, 0], X[y==1, 1], "bs", label="Iris versicolor")
plt.plot(X[y==0, 0], X[y==0, 1], "yo", label="Iris setosa")

from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])

plt.contourf(x0, x1, zz, cmap=custom_cmap)
contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
plt.clabel(contour, inline=1, fontsize=12)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 7, 0, 3.5])
```




    (0.0, 7.0, 0.0, 3.5)




![png](https://pic.imgdb.cn/item/61715e252ab3f51d91d64bb0.png)


# Support Vector Machine

![24](https://pic.imgdb.cn/item/6171569e2ab3f51d91cd3816.png)

![25](https://pic.imgdb.cn/item/6171569e2ab3f51d91cd381f.png)


```python
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"] == 2).astype(np.float64)  # Iris virginica

svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=1, loss="hinge", random_state=42)),
    ])

svm_clf.fit(X, y)
```




    Pipeline(steps=[('scaler', StandardScaler()),
                    ('linear_svc', LinearSVC(C=1, loss='hinge', random_state=42))])



![26](https://pic.imgdb.cn/item/617157662ab3f51d91ce1130.png)


```python
scaler = StandardScaler()
svm_clf1 = LinearSVC(C=1, loss="hinge", random_state=42)
svm_clf2 = LinearSVC(C=100, loss="hinge", random_state=42)

scaled_svm_clf1 = Pipeline([
        ("scaler", scaler),
        ("linear_svc", svm_clf1),
    ])
scaled_svm_clf2 = Pipeline([
        ("scaler", scaler),
        ("linear_svc", svm_clf2),
    ])

scaled_svm_clf1.fit(X, y)
scaled_svm_clf2.fit(X, y)
```

    D:\miniconda3\lib\site-packages\sklearn\svm\_base.py:986: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)





    Pipeline(steps=[('scaler', StandardScaler()),
                    ('linear_svc',
                     LinearSVC(C=100, loss='hinge', random_state=42))])




```python
# Convert to unscaled parameters
b1 = svm_clf1.decision_function([-scaler.mean_ / scaler.scale_])
b2 = svm_clf2.decision_function([-scaler.mean_ / scaler.scale_])
w1 = svm_clf1.coef_[0] / scaler.scale_
w2 = svm_clf2.coef_[0] / scaler.scale_
svm_clf1.intercept_ = np.array([b1])
svm_clf2.intercept_ = np.array([b2])
svm_clf1.coef_ = np.array([w1])
svm_clf2.coef_ = np.array([w2])

# Find support vectors (LinearSVC does not do this automatically)
t = y * 2 - 1
support_vectors_idx1 = (t * (X.dot(w1) + b1) < 1).ravel()
support_vectors_idx2 = (t * (X.dot(w2) + b2) < 1).ravel()
svm_clf1.support_vectors_ = X[support_vectors_idx1]
svm_clf2.support_vectors_ = X[support_vectors_idx2]
```


```python
def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    # At the decision boundary, w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]

    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)
```


```python
fig, axes = plt.subplots(ncols=2, figsize=(10,2.7), sharey=True)

plt.sca(axes[0])
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^", label="Iris virginica")
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs", label="Iris versicolor")
plot_svc_decision_boundary(svm_clf1, 4, 5.9)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="upper left", fontsize=14)
plt.title("$C = {}$".format(svm_clf1.C), fontsize=16)
plt.axis([4, 5.9, 0.8, 2.8])

plt.sca(axes[1])
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
plot_svc_decision_boundary(svm_clf2, 4, 5.99)
plt.xlabel("Petal length", fontsize=14)
plt.title("$C = {}$".format(svm_clf2.C), fontsize=16)
plt.axis([4, 5.9, 0.8, 2.8])
```




    (4.0, 5.9, 0.8, 2.8)




![png](https://pic.imgdb.cn/item/61715e252ab3f51d91d64bb7.png)


# Nonlinear SVM Classification


```python
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)

plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.show()
```


![png](https://pic.imgdb.cn/item/61715e252ab3f51d91d64bc0.png)



```python

from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

polynomial_svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42))
    ])

polynomial_svm_clf.fit(X, y)
```

    D:\miniconda3\lib\site-packages\sklearn\svm\_base.py:986: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)





    Pipeline(steps=[('poly_features', PolynomialFeatures(degree=3)),
                    ('scaler', StandardScaler()),
                    ('svm_clf', LinearSVC(C=10, loss='hinge', random_state=42))])




```python
def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    cs = plt.contour(x0, x1, y_decision, cmap=plt.cm.brg)
    plt.clabel(cs)


plot_predictions(polynomial_svm_clf, [-1.5, 2.5, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
```


![png](https://pic.imgdb.cn/item/61715e252ab3f51d91d64bc9.png)


## Polynomial Kernel


```python
from sklearn.svm import SVC

poly_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
    ])
poly_kernel_svm_clf.fit(X, y)


poly100_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=10, coef0=100, C=5))
    ])
poly100_kernel_svm_clf.fit(X, y)
```




    Pipeline(steps=[('scaler', StandardScaler()),
                    ('svm_clf', SVC(C=5, coef0=100, degree=10, kernel='poly'))])




```python

fig, axes = plt.subplots(ncols=2, figsize=(10.5, 4), sharey=True)

plt.sca(axes[0])
plot_predictions(poly_kernel_svm_clf, [-1.5, 2.45, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.4, -1, 1.5])
plt.title(r"$d=3, r=1, C=5$", fontsize=18)

plt.sca(axes[1])
plot_predictions(poly100_kernel_svm_clf, [-1.5, 2.45, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.4, -1, 1.5])
plt.title(r"$d=10, r=100, C=5$", fontsize=18)
plt.ylabel("")
```




    Text(0, 0.5, '')




![png](https://pic.imgdb.cn/item/61715ea02ab3f51d91d6cd50.png)


# Similarity Features

![27](https://pic.imgdb.cn/item/617157662ab3f51d91ce113a.png)

## Gaussian RBF Kernel


```python
rbf_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
    ])
rbf_kernel_svm_clf.fit(X, y)
```




    Pipeline(steps=[('scaler', StandardScaler()),
                    ('svm_clf', SVC(C=0.001, gamma=5))])




```python
from sklearn.svm import SVC

gamma1, gamma2 = 0.1, 5
C1, C2 = 0.001, 1000
hyperparams = (gamma1, C1), (gamma1, C2), (gamma2, C1), (gamma2, C2)

svm_clfs = []
for gamma, C in hyperparams:
    rbf_kernel_svm_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel="rbf", gamma=gamma, C=C))
        ])
    rbf_kernel_svm_clf.fit(X, y)
    svm_clfs.append(rbf_kernel_svm_clf)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10.5, 7), sharex=True, sharey=True)

for i, svm_clf in enumerate(svm_clfs):
    plt.sca(axes[i // 2, i % 2])
    plot_predictions(svm_clf, [-1.5, 2.45, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.45, -1, 1.5])
    gamma, C = hyperparams[i]
    plt.title(r"$\gamma = {}, C = {}$".format(gamma, C), fontsize=16)
    if i in (0, 1):
        plt.xlabel("")
    if i in (1, 3):
        plt.ylabel("")
```


![png](https://pic.imgdb.cn/item/61715ea02ab3f51d91d6cd5a.png)


As a rule of thumb, you should always try the linear kernel first (remember that `LinearSVC` is much faster than `SVC(kernel="linear"))`, especially if the training set is very large or if it has plenty of features. If the training set is not too large, you should also try the Gaussian RBF kernel; it works well in most cases.

![28](https://pic.imgdb.cn/item/617157662ab3f51d91ce1144.png)

# SVM Regression

![29](https://pic.imgdb.cn/item/617157662ab3f51d91ce114f.png)


```python
import numpy as np
np.random.seed(42)
m = 50
X = 2 * np.random.rand(m, 1)
y = (4 + 3 * X + np.random.randn(m, 1)).ravel()
```


```python
from sklearn.svm import LinearSVR

svm_reg = LinearSVR(epsilon=1.5, random_state=42)
svm_reg.fit(X, y)
```




    LinearSVR(epsilon=1.5, random_state=42)




```python
svm_reg1 = LinearSVR(epsilon=1.5, random_state=42)
svm_reg2 = LinearSVR(epsilon=0.5, random_state=42)
svm_reg1.fit(X, y)
svm_reg2.fit(X, y)

def find_support_vectors(svm_reg, X, y):
    y_pred = svm_reg.predict(X)
    off_margin = (np.abs(y - y_pred) >= svm_reg.epsilon)
    return np.argwhere(off_margin)

svm_reg1.support_ = find_support_vectors(svm_reg1, X, y)
svm_reg2.support_ = find_support_vectors(svm_reg2, X, y)

eps_x1 = 1
eps_y_pred = svm_reg1.predict([[eps_x1]])
```


```python
import matplotlib.pyplot as plt
def plot_svm_regression(svm_reg, X, y, axes):
    x1s = np.linspace(axes[0], axes[1], 100).reshape(100, 1)
    y_pred = svm_reg.predict(x1s)
    plt.plot(x1s, y_pred, "k-", linewidth=2, label=r"$\hat{y}$")
    plt.plot(x1s, y_pred + svm_reg.epsilon, "k--")
    plt.plot(x1s, y_pred - svm_reg.epsilon, "k--")
    plt.scatter(X[svm_reg.support_], y[svm_reg.support_], s=180, facecolors='#FFAAAA')
    plt.plot(X, y, "bo")
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.legend(loc="upper left", fontsize=18)
    plt.axis(axes)

fig, axes = plt.subplots(ncols=2, figsize=(9, 4), sharey=True)
plt.sca(axes[0])
plot_svm_regression(svm_reg1, X, y, [0, 2, 3, 11])
plt.title(r"$\epsilon = {}$".format(svm_reg1.epsilon), fontsize=18)
plt.ylabel(r"$y$", fontsize=18, rotation=0)
#plt.plot([eps_x1, eps_x1], [eps_y_pred, eps_y_pred - svm_reg1.epsilon], "k-", linewidth=2)
plt.annotate(
        '', xy=(eps_x1, eps_y_pred), xycoords='data',
        xytext=(eps_x1, eps_y_pred - svm_reg1.epsilon),
        textcoords='data', arrowprops={'arrowstyle': '<->', 'linewidth': 1.5}
    )
plt.text(0.91, 5.6, r"$\epsilon$", fontsize=20)
plt.sca(axes[1])
plot_svm_regression(svm_reg2, X, y, [0, 2, 3, 11])
plt.title(r"$\epsilon = {}$".format(svm_reg2.epsilon), fontsize=18)
```




    Text(0.5, 1.0, '$\\epsilon = 0.5$')




![png](https://pic.imgdb.cn/item/61715ea02ab3f51d91d6cd5f.png)


## nonlinear svm regression


```python
np.random.seed(42)
m = 100
X = 2 * np.random.rand(m, 1) - 1
y = (0.2 + 0.1 * X + 0.5 * X**2 + np.random.randn(m, 1)/10).ravel()
```


```python
from sklearn.svm import SVR

svm_poly_reg1 = SVR(kernel="poly", degree=2, C=100, epsilon=0.1, gamma="scale")
svm_poly_reg2 = SVR(kernel="poly", degree=2, C=0.01, epsilon=0.1, gamma="scale")
svm_poly_reg1.fit(X, y)
svm_poly_reg2.fit(X, y)
```




    SVR(C=0.01, degree=2, kernel='poly')




```python
fig, axes = plt.subplots(ncols=2, figsize=(9, 4), sharey=True)
plt.sca(axes[0])
plot_svm_regression(svm_poly_reg1, X, y, [-1, 1, 0, 1])
plt.title(r"$degree={}, C={}, \epsilon = {}$".format(svm_poly_reg1.degree, svm_poly_reg1.C, svm_poly_reg1.epsilon), fontsize=18)
plt.ylabel(r"$y$", fontsize=18, rotation=0)
plt.sca(axes[1])
plot_svm_regression(svm_poly_reg2, X, y, [-1, 1, 0, 1])
plt.title(r"$degree={}, C={}, \epsilon = {}$".format(svm_poly_reg2.degree, svm_poly_reg2.C, svm_poly_reg2.epsilon), fontsize=18)
```




    Text(0.5, 1.0, '$degree=2, C=0.01, \\epsilon = 0.1$')




![png](https://pic.imgdb.cn/item/61715ea02ab3f51d91d6cd69.png)


# Under the hood


```python
from sklearn import datasets
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"] == 2).astype(np.float64)  # Iris virginica
```


```python
from mpl_toolkits.mplot3d import Axes3D

def plot_3D_decision_function(ax, w, b, x1_lim=[4, 6], x2_lim=[0.8, 2.8]):
    x1_in_bounds = (X[:, 0] > x1_lim[0]) & (X[:, 0] < x1_lim[1])
    X_crop = X[x1_in_bounds]
    y_crop = y[x1_in_bounds]
    x1s = np.linspace(x1_lim[0], x1_lim[1], 20)
    x2s = np.linspace(x2_lim[0], x2_lim[1], 20)
    x1, x2 = np.meshgrid(x1s, x2s)
    xs = np.c_[x1.ravel(), x2.ravel()]
    df = (xs.dot(w) + b).reshape(x1.shape)
    m = 1 / np.linalg.norm(w)
    boundary_x2s = -x1s*(w[0]/w[1])-b/w[1]
    margin_x2s_1 = -x1s*(w[0]/w[1])-(b-1)/w[1]
    margin_x2s_2 = -x1s*(w[0]/w[1])-(b+1)/w[1]
    ax.plot_surface(x1s, x2, np.zeros_like(x1),
                    color="b", alpha=0.2, cstride=100, rstride=100)
    ax.plot(x1s, boundary_x2s, 0, "k-", linewidth=2, label=r"$h=0$")
    ax.plot(x1s, margin_x2s_1, 0, "k--", linewidth=2, label=r"$h=\pm 1$")
    ax.plot(x1s, margin_x2s_2, 0, "k--", linewidth=2)
    ax.plot(X_crop[:, 0][y_crop==1], X_crop[:, 1][y_crop==1], 0, "g^")
    ax.plot_wireframe(x1, x2, df, alpha=0.3, color="k")
    ax.plot(X_crop[:, 0][y_crop==0], X_crop[:, 1][y_crop==0], 0, "bs")
    ax.axis(x1_lim + x2_lim)
    ax.text(4.5, 2.5, 3.8, "Decision function $h$", fontsize=16)
    ax.set_xlabel(r"Petal length", fontsize=16, labelpad=10)
    ax.set_ylabel(r"Petal width", fontsize=16, labelpad=10)
    ax.set_zlabel(r"$h = \mathbf{w}^T \mathbf{x} + b$", fontsize=18, labelpad=5)
    ax.legend(loc="upper left", fontsize=16)



fig = plt.figure(figsize=(11, 6))
ax1 = fig.add_subplot(111, projection='3d')
plot_3D_decision_function(ax1, w=svm_clf2.coef_[0], b=svm_clf2.intercept_[0])
```


![png](https://pic.imgdb.cn/item/61715ea02ab3f51d91d6cd79.png)


![30](https://pic.imgdb.cn/item/617157662ab3f51d91ce1157.png)

![31](https://pic.imgdb.cn/item/617157a22ab3f51d91ce5460.png)

The dual problem is faster to solve than the primal one when the number of training instances is smaller than the number of features. More importantly, the dual problem makes the kernel trick possible, while the primal does not.


![32](https://pic.imgdb.cn/item/617157a22ab3f51d91ce5465.png)

## hinge loss


```python
t = np.linspace(-2, 4, 200)
h = np.where(1 - t < 0, 0, 1 - t)  # max(0, 1-t)

plt.figure(figsize=(5,2.8))
plt.plot(t, h, "b-", linewidth=2, label="$max(0, 1 - t)$")
plt.grid(True, which='both')
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.yticks(np.arange(-1, 2.5, 1))
plt.xlabel("$t$", fontsize=16)
plt.axis([-2, 4, -1, 2.5])
plt.legend(loc="upper right", fontsize=16)
```




    <matplotlib.legend.Legend at 0x1f6967814c8>




![png](https://pic.imgdb.cn/item/61715ee52ab3f51d91d71c0a.png)


## Linear SVM classifier implementation using Batch Gradient Descent

![33](https://pic.imgdb.cn/item/617157a22ab3f51d91ce546c.png)


```python
# Training set
X = iris["data"][:, (2, 3)] # petal length, petal width
y = (iris["target"] == 2).astype(np.float64).reshape(-1, 1) # Iris virginica
```


```python
from sklearn.base import BaseEstimator

class MyLinearSVC(BaseEstimator):
    def __init__(self, C=1, eta0=1, eta_d=10000, n_epochs=1000, random_state=None):
        self.C = C
        self.eta0 = eta0
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.eta_d = eta_d

    def eta(self, epoch):
        return self.eta0 / (epoch + self.eta_d)
        
    def fit(self, X, y):
        # Random initialization
        if self.random_state:
            np.random.seed(self.random_state)
        w = np.random.randn(X.shape[1], 1) # n feature weights
        b = 0

        m = len(X)
        t = y * 2 - 1  # -1 if t==0, +1 if t==1
        X_t = X * t
        self.Js=[]

        # Training
        for epoch in range(self.n_epochs):
            support_vectors_idx = (X_t.dot(w) + t * b < 1).ravel()
            X_t_sv = X_t[support_vectors_idx]
            t_sv = t[support_vectors_idx]

            J = 1/2 * np.sum(w * w) + self.C * (np.sum(1 - X_t_sv.dot(w)) - b * np.sum(t_sv))
            self.Js.append(J)

            w_gradient_vector = w - self.C * np.sum(X_t_sv, axis=0).reshape(-1, 1)
            b_derivative = -self.C * np.sum(t_sv)
                
            w = w - self.eta(epoch) * w_gradient_vector
            b = b - self.eta(epoch) * b_derivative
            

        self.intercept_ = np.array([b])
        self.coef_ = np.array([w])
        support_vectors_idx = (X_t.dot(w) + t * b < 1).ravel()
        self.support_vectors_ = X[support_vectors_idx]
        return self

    def decision_function(self, X):
        return X.dot(self.coef_[0]) + self.intercept_[0]

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(np.float64)

C=2
svm_clf = MyLinearSVC(C=C, eta0 = 10, eta_d = 1000, n_epochs=60000, random_state=2)
svm_clf.fit(X, y)
svm_clf.predict(np.array([[5, 2], [4, 1]]))
```




    array([[1.],
           [0.]])




```python
plt.plot(range(svm_clf.n_epochs), svm_clf.Js)
plt.axis([0, svm_clf.n_epochs, 0, 100])
```




    (0.0, 60000.0, 0.0, 100.0)




![png](https://pic.imgdb.cn/item/61715ee52ab3f51d91d71c0f.png)



```python
from sklearn.svm import SVC 
svm_clf2 = SVC(kernel="linear", C=C)
svm_clf2.fit(X, y.ravel())
```




    SVC(C=2, kernel='linear')




```python
yr = y.ravel()
fig, axes = plt.subplots(ncols=2, figsize=(11, 3.2), sharey=True)
plt.sca(axes[0])
plt.plot(X[:, 0][yr==1], X[:, 1][yr==1], "g^", label="Iris virginica")
plt.plot(X[:, 0][yr==0], X[:, 1][yr==0], "bs", label="Not Iris virginica")
plot_svc_decision_boundary(svm_clf, 4, 6)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.title("MyLinearSVC", fontsize=14)
plt.axis([4, 6, 0.8, 2.8])
plt.legend(loc="upper left")

plt.sca(axes[1])
plt.plot(X[:, 0][yr==1], X[:, 1][yr==1], "g^")
plt.plot(X[:, 0][yr==0], X[:, 1][yr==0], "bs")
plot_svc_decision_boundary(svm_clf2, 4, 6)
plt.xlabel("Petal length", fontsize=14)
plt.title("SVC", fontsize=14)
plt.axis([4, 6, 0.8, 2.8])
```




    (4.0, 6.0, 0.8, 2.8)




![png](https://pic.imgdb.cn/item/61715ee52ab3f51d91d71c18.png)


# Decision Trees


```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data[:, 2:] # petal length and width
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X, y)
```




    DecisionTreeClassifier(max_depth=2, random_state=42)



![34](https://pic.imgdb.cn/item/617157a22ab3f51d91ce5477.png)


```python
IMAGES_PATH = './'
# from graphviz import Source
from sklearn.tree import export_graphviz

export_graphviz(
        tree_clf,
        out_file=os.path.join(IMAGES_PATH, "iris_tree.dot"),
        feature_names=iris.feature_names[2:],
        class_names=iris.target_names,
        rounded=True,
        filled=True
    )

# Source.from_file(os.path.join(IMAGES_PATH, "iris_tree.dot"))
```

![35](https://pic.imgdb.cn/item/617157a22ab3f51d91ce5482.png)


```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_boundary(clf, X, y, axes=[0, 7.5, 0, 3], iris=True, legend=False, plot_training=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if not iris:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    if plot_training:
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris setosa")
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris versicolor")
        plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="Iris virginica")
        plt.axis(axes)
    if iris:
        plt.xlabel("Petal length", fontsize=14)
        plt.ylabel("Petal width", fontsize=14)
    else:
        plt.xlabel(r"$x_1$", fontsize=18)
        plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    if legend:
        plt.legend(loc="lower right", fontsize=14)

plt.figure(figsize=(8, 4))
plot_decision_boundary(tree_clf, X, y)
plt.plot([2.45, 2.45], [0, 3], "k-", linewidth=2)
plt.plot([2.45, 7.5], [1.75, 1.75], "k--", linewidth=2)
plt.plot([4.95, 4.95], [0, 1.75], "k:", linewidth=2)
plt.plot([4.85, 4.85], [1.75, 3], "k:", linewidth=2)
plt.text(1.40, 1.0, "Depth=0", fontsize=15)
plt.text(3.2, 1.80, "Depth=1", fontsize=13)
plt.text(4.05, 0.5, "(Depth=2)", fontsize=11)
```




    Text(4.05, 0.5, '(Depth=2)')




![png](https://pic.imgdb.cn/item/61715ee52ab3f51d91d71c1f.png)



```python
tree_clf.predict_proba([[5, 1.5]])
```




    array([[0.        , 0.90740741, 0.09259259]])




```python
tree_clf.predict([[5, 1.5]])
```




    array([1])



![36](https://pic.imgdb.cn/item/617158042ab3f51d91cebb40.png)

![37](https://pic.imgdb.cn/item/617158042ab3f51d91cebb48.png)

![38](https://pic.imgdb.cn/item/617158042ab3f51d91cebb50.png)

# Regularization Hyperparameters

![39](https://pic.imgdb.cn/item/617158042ab3f51d91cebb5a.jpg)


```python
from sklearn.datasets import make_moons
Xm, ym = make_moons(n_samples=100, noise=0.25, random_state=53)

deep_tree_clf1 = DecisionTreeClassifier(random_state=42)
deep_tree_clf2 = DecisionTreeClassifier(min_samples_leaf=4, random_state=42)
deep_tree_clf1.fit(Xm, ym)
deep_tree_clf2.fit(Xm, ym)

fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
plt.sca(axes[0])
plot_decision_boundary(deep_tree_clf1, Xm, ym, axes=[-1.5, 2.4, -1, 1.5], iris=False)
plt.title("No restrictions", fontsize=16)
plt.sca(axes[1])
plot_decision_boundary(deep_tree_clf2, Xm, ym, axes=[-1.5, 2.4, -1, 1.5], iris=False)
plt.title("min_samples_leaf = {}".format(deep_tree_clf2.min_samples_leaf), fontsize=14)
plt.ylabel("")
```




    Text(0, 0.5, '')




![png](https://pic.imgdb.cn/item/61715ee52ab3f51d91d71c2b.png)


# Regression Trees

![40](https://pic.imgdb.cn/item/617158042ab3f51d91cebb65.png)


```python
# Quadratic training set + noise
np.random.seed(42)
m = 200
X = np.random.rand(m, 1)
y = 4 * (X - 0.5) ** 2
y = y + np.random.randn(m, 1) / 10
```


```python
from sklearn.tree import DecisionTreeRegressor

tree_reg1 = DecisionTreeRegressor(random_state=42, max_depth=2)
tree_reg2 = DecisionTreeRegressor(random_state=42, max_depth=3)
tree_reg1.fit(X, y)
tree_reg2.fit(X, y)

def plot_regression_predictions(tree_reg, X, y, axes=[0, 1, -0.2, 1], ylabel="$y$"):
    x1 = np.linspace(axes[0], axes[1], 500).reshape(-1, 1)
    y_pred = tree_reg.predict(x1)
    plt.axis(axes)
    plt.xlabel("$x_1$", fontsize=18)
    if ylabel:
        plt.ylabel(ylabel, fontsize=18, rotation=0)
    plt.plot(X, y, "b.")
    plt.plot(x1, y_pred, "r.-", linewidth=2, label=r"$\hat{y}$")

fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
plt.sca(axes[0])
plot_regression_predictions(tree_reg1, X, y)
for split, style in ((0.1973, "k-"), (0.0917, "k--"), (0.7718, "k--")):
    plt.plot([split, split], [-0.2, 1], style, linewidth=2)
plt.text(0.21, 0.65, "Depth=0", fontsize=15)
plt.text(0.01, 0.2, "Depth=1", fontsize=13)
plt.text(0.65, 0.8, "Depth=1", fontsize=13)
plt.legend(loc="upper center", fontsize=18)
plt.title("max_depth=2", fontsize=14)

plt.sca(axes[1])
plot_regression_predictions(tree_reg2, X, y, ylabel=None)
for split, style in ((0.1973, "k-"), (0.0917, "k--"), (0.7718, "k--")):
    plt.plot([split, split], [-0.2, 1], style, linewidth=2)
for split in (0.0458, 0.1298, 0.2873, 0.9040):
    plt.plot([split, split], [-0.2, 1], "k:", linewidth=1)
plt.text(0.3, 0.5, "Depth=2", fontsize=13)
plt.title("max_depth=3", fontsize=14)
```




    Text(0.5, 1.0, 'max_depth=3')




![png](https://pic.imgdb.cn/item/61715f412ab3f51d91d78ae5.png)



```python
IMAGES_PATH = './'
export_graphviz(
        tree_reg1,
        out_file=os.path.join(IMAGES_PATH, "regression_tree.dot"),
        feature_names=["x1"],
        rounded=True,
        filled=True
    )
```


```python
tree_reg1 = DecisionTreeRegressor(random_state=42)
tree_reg2 = DecisionTreeRegressor(random_state=42, min_samples_leaf=10)
tree_reg1.fit(X, y)
tree_reg2.fit(X, y)

x1 = np.linspace(0, 1, 500).reshape(-1, 1)
y_pred1 = tree_reg1.predict(x1)
y_pred2 = tree_reg2.predict(x1)

fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)

plt.sca(axes[0])
plt.plot(X, y, "b.")
plt.plot(x1, y_pred1, "r.-", linewidth=2, label=r"$\hat{y}$")
plt.axis([0, 1, -0.2, 1.1])
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", fontsize=18, rotation=0)
plt.legend(loc="upper center", fontsize=18)
plt.title("No restrictions", fontsize=14)

plt.sca(axes[1])
plt.plot(X, y, "b.")
plt.plot(x1, y_pred2, "r.-", linewidth=2, label=r"$\hat{y}$")
plt.axis([0, 1, -0.2, 1.1])
plt.xlabel("$x_1$", fontsize=18)
plt.title("min_samples_leaf={}".format(tree_reg2.min_samples_leaf), fontsize=14)
```




    Text(0.5, 1.0, 'min_samples_leaf=10')




![png](https://pic.imgdb.cn/item/61715f412ab3f51d91d78aea.png)


# Instability

![41](https://pic.imgdb.cn/item/6171582e2ab3f51d91cee60b.png)


```python
# senstive to rotation
np.random.seed(6)
Xs = np.random.rand(100, 2) - 0.5
ys = (Xs[:, 0] > 0).astype(np.float32) * 2

angle = np.pi / 4
rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
Xsr = Xs.dot(rotation_matrix)

tree_clf_s = DecisionTreeClassifier(random_state=42)
tree_clf_s.fit(Xs, ys)
tree_clf_sr = DecisionTreeClassifier(random_state=42)
tree_clf_sr.fit(Xsr, ys)

fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
plt.sca(axes[0])
plot_decision_boundary(tree_clf_s, Xs, ys, axes=[-0.7, 0.7, -0.7, 0.7], iris=False)
plt.sca(axes[1])
plot_decision_boundary(tree_clf_sr, Xsr, ys, axes=[-0.7, 0.7, -0.7, 0.7], iris=False)
plt.ylabel("")

```




    Text(0, 0.5, '')




![png](https://pic.imgdb.cn/item/61715f412ab3f51d91d78af3.png)


We've seen that small changes in the dataset (such as a rotation) may produce a very different Decision Tree. Now let's show that training the same model on the same data may produce a very different model every time, since the CART training algorithm used by Scikit-Learn is stochastic. To show this, we will set random_state to a different value than earlier:


```python
iris = load_iris()
X = iris.data[:, 2:] # petal length and width
y = iris.target

tree_clf_tweaked = DecisionTreeClassifier(max_depth=2, random_state=40)
tree_clf_tweaked.fit(X, y)

plt.figure(figsize=(8, 4))
plot_decision_boundary(tree_clf_tweaked, X, y, legend=False)
plt.plot([0, 7.5], [0.8, 0.8], "k-", linewidth=2)
plt.plot([0, 7.5], [1.75, 1.75], "k--", linewidth=2)
plt.text(1.0, 0.9, "Depth=0", fontsize=15)
plt.text(1.0, 1.80, "Depth=1", fontsize=13)
```




    Text(1.0, 1.8, 'Depth=1')




![png](https://pic.imgdb.cn/item/61715f412ab3f51d91d78afc.png)


# Ensemble Learning and Random Forests

![42](https://pic.imgdb.cn/item/6171582e2ab3f51d91cee610.png)

Suppose you have a slightly biased coin that has a 51% chance of coming up heads and 49% chance of coming up tails. If you toss it 1,000 times, you will generally get more or less 510 heads and 490 tails, and hence a majority of heads. If you do the math, you will find that the probability of obtaining a majority of heads after 1,000 tosses is close to 75%. The more you toss the coin, the higher the probability (e.g., with 10,000 tosses, the probability climbs over 97%).


```python
from scipy.stats import binom

n = 1000  # tosses
p = 0.51 # probability of head (coin is biased)
x = n / 2 + 1  # to get majority of heads out of n tosses I need just n / 2 + 1
print(f'n={n}, prob of a majority of heads={1 - binom.cdf(x - 1, n, p)}') 

n = 10000
x = n / 2 + 1
print(f'n={n}, prob of a majority of heads={1 - binom.cdf(x - 1, n, p)}') 

```

    n=1000, prob of a majority of heads=0.7260985557303354
    n=10000, prob of a majority of heads=0.9767182874807615



```python
import numpy as np
np.random.seed(3)
heads_proba = 0.51
coin_tosses = (np.random.rand(10000, 10) < heads_proba).astype(np.int32)
cumulative_heads_ratio = np.cumsum(coin_tosses, axis=0) / np.arange(1, 10001).reshape(-1, 1)
```


```python
import matplotlib.pyplot as plt
plt.figure(figsize=(8,3.5))
plt.plot(cumulative_heads_ratio)
plt.plot([0, 10000], [0.51, 0.51], "k--", linewidth=2, label="51%")
plt.plot([0, 10000], [0.5, 0.5], "k-", label="50%")
plt.xlabel("Number of coin tosses")
plt.ylabel("Heads ratio")
plt.legend(loc="lower right")
plt.axis([0, 10000, 0.42, 0.58])
```




    (0.0, 10000.0, 0.42, 0.58)




![png](https://pic.imgdb.cn/item/61715f412ab3f51d91d78b05.png)


![43](https://pic.imgdb.cn/item/6171582e2ab3f51d91cee621.png)

# Voting Classifiers


```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```

Note: to be future-proof, we set solver="lbfgs", n_estimators=100, and gamma="scale" since these will be the default values in upcoming Scikit-Learn versions.


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

log_clf = LogisticRegression(solver="lbfgs", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
svm_clf = SVC(gamma="scale", random_state=42)

voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
                                voting='hard')
```


```python
voting_clf.fit(X_train, y_train)
```




    VotingClassifier(estimators=[('lr', LogisticRegression(random_state=42)),
                                 ('rf', RandomForestClassifier(random_state=42)),
                                 ('svc', SVC(random_state=42))])




```python
from sklearn.metrics import accuracy_score

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
```

    LogisticRegression 0.864
    RandomForestClassifier 0.896
    SVC 0.896
    VotingClassifier 0.912


![44](https://pic.imgdb.cn/item/6171582e2ab3f51d91cee628.png)


```python
log_clf = LogisticRegression(solver="lbfgs", random_state=42)
rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
svm_clf = SVC(gamma="scale", probability=True, random_state=42)

voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
                                voting='soft')
```


```python
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
```

    LogisticRegression 0.864
    RandomForestClassifier 0.896
    SVC 0.896
    VotingClassifier 0.92


# Bagging and Pasting

![45](https://pic.imgdb.cn/item/6171582e2ab3f51d91cee62f.png)


```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, max_samples=100, bootstrap=True, random_state=42, n_jobs=-1) # –1 tells Scikit-Learn to use all available cores
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
```


```python
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
```




    0.904




```python
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)
print(accuracy_score(y_test, y_pred_tree))
```

    0.856



```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.45, -1, 1.5], alpha=0.5, contour=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", alpha=alpha)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
```


```python
fix, axes = plt.subplots(ncols=2, figsize=(10,4), sharey=True)
plt.sca(axes[0])
plot_decision_boundary(tree_clf, X, y)
plt.title("Decision Tree", fontsize=14)
plt.sca(axes[1])
plot_decision_boundary(bag_clf, X, y)
plt.title("Decision Trees with Bagging", fontsize=14)
plt.ylabel("")
```




    Text(0, 0.5, '')




![png](https://pic.imgdb.cn/item/61715f802ab3f51d91d7cb52.png)


![46](https://pic.imgdb.cn/item/617158652ab3f51d91cf1ea7.png)

# Out-of-bag Evaluation 

![47](https://pic.imgdb.cn/item/617158652ab3f51d91cf1eaf.jpg)


```python
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500,
    bootstrap=True, oob_score=True, random_state=40)
bag_clf.fit(X_train, y_train)
bag_clf.oob_score_
```




    0.8986666666666666




```python
bag_clf.oob_decision_function_
```




    array([[0.32275132, 0.67724868],
           [0.34117647, 0.65882353],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.09497207, 0.90502793],
           [0.31147541, 0.68852459],
           [0.01754386, 0.98245614],
           [0.97109827, 0.02890173],
           [0.97765363, 0.02234637],
           [0.74404762, 0.25595238],
           [0.        , 1.        ],
           [0.7173913 , 0.2826087 ],
           [0.85026738, 0.14973262],
           [0.97222222, 0.02777778],
           [0.0625    , 0.9375    ],
           [0.        , 1.        ],
           [0.97837838, 0.02162162],
           [0.94642857, 0.05357143],
           [1.        , 0.        ],
           [0.01704545, 0.98295455],
           [0.39473684, 0.60526316],
           [0.88700565, 0.11299435],
           [1.        , 0.        ],
           [0.97790055, 0.02209945],
           [0.        , 1.        ],
           [0.99428571, 0.00571429],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.62569832, 0.37430168],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.13402062, 0.86597938],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.38251366, 0.61748634],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.27093596, 0.72906404],
           [0.34146341, 0.65853659],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.00531915, 0.99468085],
           [0.98843931, 0.01156069],
           [0.91428571, 0.08571429],
           [0.97282609, 0.02717391],
           [0.98019802, 0.01980198],
           [0.        , 1.        ],
           [0.07361963, 0.92638037],
           [0.98019802, 0.01980198],
           [0.0052356 , 0.9947644 ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.97790055, 0.02209945],
           [0.8       , 0.2       ],
           [0.42424242, 0.57575758],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.66477273, 0.33522727],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [0.86781609, 0.13218391],
           [1.        , 0.        ],
           [0.56725146, 0.43274854],
           [0.1576087 , 0.8423913 ],
           [0.66492147, 0.33507853],
           [0.91709845, 0.08290155],
           [0.        , 1.        ],
           [0.16759777, 0.83240223],
           [0.87434555, 0.12565445],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.995     , 0.005     ],
           [0.        , 1.        ],
           [0.07878788, 0.92121212],
           [0.05418719, 0.94581281],
           [0.29015544, 0.70984456],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.83040936, 0.16959064],
           [0.01092896, 0.98907104],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.21465969, 0.78534031],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.94660194, 0.05339806],
           [0.77094972, 0.22905028],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.16574586, 0.83425414],
           [0.65306122, 0.34693878],
           [0.        , 1.        ],
           [0.02564103, 0.97435897],
           [0.50555556, 0.49444444],
           [1.        , 0.        ],
           [0.03208556, 0.96791444],
           [0.99435028, 0.00564972],
           [0.23699422, 0.76300578],
           [0.49509804, 0.50490196],
           [0.9947644 , 0.0052356 ],
           [0.00555556, 0.99444444],
           [0.98963731, 0.01036269],
           [0.26153846, 0.73846154],
           [0.92972973, 0.07027027],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.80113636, 0.19886364],
           [1.        , 0.        ],
           [0.0106383 , 0.9893617 ],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [0.98181818, 0.01818182],
           [1.        , 0.        ],
           [0.01036269, 0.98963731],
           [0.97752809, 0.02247191],
           [0.99453552, 0.00546448],
           [0.01960784, 0.98039216],
           [0.17857143, 0.82142857],
           [0.98387097, 0.01612903],
           [0.29533679, 0.70466321],
           [0.98295455, 0.01704545],
           [0.        , 1.        ],
           [0.00561798, 0.99438202],
           [0.75690608, 0.24309392],
           [0.38624339, 0.61375661],
           [0.40625   , 0.59375   ],
           [0.87368421, 0.12631579],
           [0.92462312, 0.07537688],
           [0.05181347, 0.94818653],
           [0.82802548, 0.17197452],
           [0.01546392, 0.98453608],
           [0.        , 1.        ],
           [0.02298851, 0.97701149],
           [0.9726776 , 0.0273224 ],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [0.01041667, 0.98958333],
           [0.        , 1.        ],
           [0.03804348, 0.96195652],
           [0.02040816, 0.97959184],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [0.94915254, 0.05084746],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [0.99462366, 0.00537634],
           [0.        , 1.        ],
           [0.39378238, 0.60621762],
           [0.33152174, 0.66847826],
           [0.00609756, 0.99390244],
           [0.        , 1.        ],
           [0.3172043 , 0.6827957 ],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.00588235, 0.99411765],
           [0.        , 1.        ],
           [0.98924731, 0.01075269],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.62893082, 0.37106918],
           [0.92344498, 0.07655502],
           [0.        , 1.        ],
           [0.99526066, 0.00473934],
           [1.        , 0.        ],
           [0.98888889, 0.01111111],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.06989247, 0.93010753],
           [1.        , 0.        ],
           [0.03608247, 0.96391753],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.02185792, 0.97814208],
           [1.        , 0.        ],
           [0.95808383, 0.04191617],
           [0.78362573, 0.21637427],
           [0.56650246, 0.43349754],
           [0.        , 1.        ],
           [0.18023256, 0.81976744],
           [1.        , 0.        ],
           [0.93121693, 0.06878307],
           [0.97175141, 0.02824859],
           [1.        , 0.        ],
           [0.00531915, 0.99468085],
           [0.        , 1.        ],
           [0.43010753, 0.56989247],
           [0.85858586, 0.14141414],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.00558659, 0.99441341],
           [0.        , 1.        ],
           [0.96923077, 0.03076923],
           [0.        , 1.        ],
           [0.21649485, 0.78350515],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.98477157, 0.01522843],
           [0.8       , 0.2       ],
           [0.99441341, 0.00558659],
           [0.        , 1.        ],
           [0.09497207, 0.90502793],
           [0.99492386, 0.00507614],
           [0.01714286, 0.98285714],
           [0.        , 1.        ],
           [0.02747253, 0.97252747],
           [1.        , 0.        ],
           [0.77005348, 0.22994652],
           [0.        , 1.        ],
           [0.90229885, 0.09770115],
           [0.98387097, 0.01612903],
           [0.22222222, 0.77777778],
           [0.20348837, 0.79651163],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.20338983, 0.79661017],
           [0.98181818, 0.01818182],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.98969072, 0.01030928],
           [0.        , 1.        ],
           [0.48663102, 0.51336898],
           [1.        , 0.        ],
           [0.00529101, 0.99470899],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.08379888, 0.91620112],
           [0.12352941, 0.87647059],
           [0.99415205, 0.00584795],
           [0.03517588, 0.96482412],
           [1.        , 0.        ],
           [0.39790576, 0.60209424],
           [0.05434783, 0.94565217],
           [0.53191489, 0.46808511],
           [0.51898734, 0.48101266],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.60869565, 0.39130435],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.24157303, 0.75842697],
           [0.81578947, 0.18421053],
           [0.08717949, 0.91282051],
           [0.99453552, 0.00546448],
           [0.82142857, 0.17857143],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [0.11904762, 0.88095238],
           [0.04188482, 0.95811518],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.89150943, 0.10849057],
           [0.19230769, 0.80769231],
           [0.95238095, 0.04761905],
           [0.00515464, 0.99484536],
           [0.59375   , 0.40625   ],
           [0.07692308, 0.92307692],
           [0.99484536, 0.00515464],
           [0.83684211, 0.16315789],
           [0.        , 1.        ],
           [0.99484536, 0.00515464],
           [0.95360825, 0.04639175],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.26395939, 0.73604061],
           [0.98461538, 0.01538462],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.00574713, 0.99425287],
           [0.85142857, 0.14857143],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.75301205, 0.24698795],
           [0.8969697 , 0.1030303 ],
           [1.        , 0.        ],
           [0.75555556, 0.24444444],
           [0.48863636, 0.51136364],
           [0.        , 1.        ],
           [0.92473118, 0.07526882],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.87709497, 0.12290503],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [0.74752475, 0.25247525],
           [0.09146341, 0.90853659],
           [0.42268041, 0.57731959],
           [0.22395833, 0.77604167],
           [0.        , 1.        ],
           [0.87046632, 0.12953368],
           [0.78212291, 0.21787709],
           [0.00507614, 0.99492386],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.02884615, 0.97115385],
           [0.96      , 0.04      ],
           [0.93478261, 0.06521739],
           [1.        , 0.        ],
           [0.50731707, 0.49268293],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.01604278, 0.98395722],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.96987952, 0.03012048],
           [0.        , 1.        ],
           [0.05172414, 0.94827586],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.99494949, 0.00505051],
           [0.01675978, 0.98324022],
           [1.        , 0.        ],
           [0.14583333, 0.85416667],
           [0.        , 1.        ],
           [0.00546448, 0.99453552],
           [0.        , 1.        ],
           [0.41836735, 0.58163265],
           [0.13095238, 0.86904762],
           [0.22110553, 0.77889447],
           [1.        , 0.        ],
           [0.97647059, 0.02352941],
           [0.21195652, 0.78804348],
           [0.98882682, 0.01117318],
           [0.        , 1.        ],
           [0.        , 1.        ],
           [1.        , 0.        ],
           [0.96428571, 0.03571429],
           [0.34554974, 0.65445026],
           [0.98235294, 0.01764706],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.99465241, 0.00534759],
           [0.        , 1.        ],
           [0.06043956, 0.93956044],
           [0.98214286, 0.01785714],
           [1.        , 0.        ],
           [0.03108808, 0.96891192],
           [0.58854167, 0.41145833]])



![48](https://pic.imgdb.cn/item/617158652ab3f51d91cf1eb8.png)

# Random Forest


```python
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)
rnd_clf.fit(X_train, y_train)
y_pred = rnd_clf.predict(X_test)
accuracy_score(y_test, y_pred)
```




    0.912




```python
# Extra-Trees

from sklearn.ensemble import ExtraTreesClassifier
ext_clf = ExtraTreesClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)
ext_clf.fit(X_train, y_train)
y_pred = ext_clf.predict(X_test)
accuracy_score(y_test, y_pred)

```




    0.912



# Feature Importance

![49](https://pic.imgdb.cn/item/617158652ab3f51d91cf1ec3.png)


```python
from sklearn.datasets import load_iris
iris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators=500, random_state=42)
rnd_clf.fit(iris["data"], iris["target"])
for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
    print(name, score)
```

    sepal length (cm) 0.11249225099876375
    sepal width (cm) 0.02311928828251033
    petal length (cm) 0.4410304643639577
    petal width (cm) 0.4233579963547682


Warning: since Scikit-Learn 0.24, fetch_openml() returns a Pandas DataFrame by default. To avoid this and keep the same code as in the book, we use as_frame=False.


```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
```


```python
mnist.target = mnist.target.astype(np.uint8)
```


```python
rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rnd_clf.fit(mnist["data"], mnist["target"])
```




    RandomForestClassifier(random_state=42)




```python
def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = plt.cm.hot,
               interpolation="nearest")
    plt.axis("off")


plot_digit(rnd_clf.feature_importances_)

cbar = plt.colorbar(ticks=[rnd_clf.feature_importances_.min(), rnd_clf.feature_importances_.max()])
cbar.ax.set_yticklabels(['Not important', 'Very important'])
```




    [Text(1, 0.0, 'Not important'),
     Text(1, 0.009791489757332336, 'Very important')]




![png](https://pic.imgdb.cn/item/61715f802ab3f51d91d7cb58.png)


# Boosting

**Boosting** (originally called **hypothesis boosting**) refers to any Ensemble method that can combine several weak learners into a strong learner. The general idea of most boosting methods is to train predictors sequentially, each trying to correct its predecessor.

![50](https://pic.imgdb.cn/item/617158652ab3f51d91cf1ece.png)


```python
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200, algorithm='SAMME.R', learning_rate=0.5, random_state=42)
ada_clf.fit(X_train, y_train)
```




    AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1),
                       learning_rate=0.5, n_estimators=200, random_state=42)




```python
plot_decision_boundary(ada_clf, X, y)
```


![png](https://pic.imgdb.cn/item/61715f802ab3f51d91d7cb61.png)



```python
from sklearn.svm import SVC
m = len(X_train)

fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
for subplot, learning_rate in ((0, 1), (1, 0.5)):
    sample_weights = np.ones(m) / m
    plt.sca(axes[subplot])
    for i in range(5):
        svm_clf = SVC(kernel='rbf', C=0.2, gamma=0.6, random_state=42)
        svm_clf.fit(X_train, y_train, sample_weight=sample_weights * m)
        y_pred = svm_clf.predict(X_train)

        r = sample_weights[y_pred != y_train].sum() / sample_weights.sum()   # equation 7-1
        alpha = learning_rate * np.log((1 - r) / r)                         # equation 7-2
        sample_weights[y_pred != y_train] *= np.exp(alpha)                   # equation 7-3
        sample_weights /= sample_weights.sum()                              # normalization step

        plot_decision_boundary(svm_clf, X, y, alpha=0.2)
        plt.title('learning rate = {}'.format(learning_rate), fontsize=16)

    if subplot == 0:
        plt.text(-0.75, -0.95, "1", fontsize=14)
        plt.text(-1.05, -0.95, "2", fontsize=14)
        plt.text(1.0, -0.95, "3", fontsize=14)
        plt.text(-1.45, -0.5, "4", fontsize=14)
        plt.text(1.36,  -0.95, "5", fontsize=14)
    else:
        plt.ylabel("")

```


![png](https://pic.imgdb.cn/item/61715f802ab3f51d91d7cb6b.png)


## Gradient Boosting

![51](https://pic.imgdb.cn/item/617159822ab3f51d91d14a7f.png)


```python
np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3*X[:, 0]**2 + 0.05 * np.random.randn(100)
```


```python
from sklearn.tree import DecisionTreeRegressor

tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg1.fit(X, y)

y2 = y - tree_reg1.predict(X)
tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg2.fit(X, y2)

y3 = y2 - tree_reg2.predict(X)
tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg3.fit(X, y3)
```




    DecisionTreeRegressor(max_depth=2, random_state=42)




```python
X_new = np.array([[0.8]])

y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))

y_pred
```




    array([0.75026781])




```python
def plot_predictions(regressors, X, y, axes, label=None, style="r-", data_style="b.", data_label=None):
    x1 = np.linspace(axes[0], axes[1], 500)
    y_pred = sum(regressor.predict(x1.reshape(-1, 1)) for regressor in regressors)
    plt.plot(X[:, 0], y, data_style, label=data_label)
    plt.plot(x1, y_pred, style, linewidth=2, label=label)
    if label or data_label:
        plt.legend(loc="upper center", fontsize=16)
    plt.axis(axes)
```


```python
plt.figure(figsize=(11,11))

plt.subplot(321)
plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h_1(x_1)$", style="g-", data_label="Training set")
plt.ylabel("$y$", fontsize=16, rotation=0)
plt.title("Residuals and tree predictions", fontsize=16)

plt.subplot(322)
plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1)$", data_label="Training set")
plt.ylabel("$y$", fontsize=16, rotation=0)
plt.title("Ensemble predictions", fontsize=16)

plt.subplot(323)
plot_predictions([tree_reg2], X, y2, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_2(x_1)$", style="g-", data_style="k+", data_label="Residuals")
plt.ylabel("$y - h_1(x_1)$", fontsize=16)

plt.subplot(324)
plot_predictions([tree_reg1, tree_reg2], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1)$")
plt.ylabel("$y$", fontsize=16, rotation=0)

plt.subplot(325)
plot_predictions([tree_reg3], X, y3, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_3(x_1)$", style="g-", data_style="k+")
plt.ylabel("$y - h_1(x_1) - h_2(x_1)$", fontsize=16)
plt.xlabel("$x_1$", fontsize=16)

plt.subplot(326)
plot_predictions([tree_reg1, tree_reg2, tree_reg3], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1) + h_3(x_1)$")
plt.xlabel("$x_1$", fontsize=16)
plt.ylabel("$y$", fontsize=16, rotation=0)

```




    Text(0, 0.5, '$y$')




![png](https://pic.imgdb.cn/item/61715f802ab3f51d91d7cb79.png)


A simpler way to train GBRT ensembles is to use Scikit-Learn’s `GradientBoostingRegressor` class. Much like the `RandomForestRegressor`
class, it has hyperparameters to control the growth of Decision Trees (e.g., `max_depth`, `min_samples_leaf`), as well as hyperparameters to control the ensemble training, such as the number of trees (n_estimators).


```python
from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0, random_state=42)
gbrt.fit(X, y)

gbrt_slow = GradientBoostingRegressor(max_depth=2, n_estimators=200, learning_rate=0.1, random_state=42)
gbrt_slow.fit(X, y)
```




    GradientBoostingRegressor(max_depth=2, n_estimators=200, random_state=42)



![52](https://pic.imgdb.cn/item/617159822ab3f51d91d14a8b.png)


```python
fig, axes = plt.subplots(ncols=2, figsize=(10,4), sharey=True)

plt.sca(axes[0])
plot_predictions([gbrt], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="Ensemble predictions")
plt.title("learning_rate={}, n_estimators={}".format(gbrt.learning_rate, gbrt.n_estimators), fontsize=14)
plt.xlabel("$x_1$", fontsize=16)
plt.ylabel("$y$", fontsize=16, rotation=0)

plt.sca(axes[1])
plot_predictions([gbrt_slow], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
plt.title("learning_rate={}, n_estimators={}".format(gbrt_slow.learning_rate, gbrt_slow.n_estimators), fontsize=14)
plt.xlabel("$x_1$", fontsize=16)
```




    Text(0.5, 0, '$x_1$')




![png](https://pic.imgdb.cn/item/61715fca2ab3f51d91d81c4f.png)


In order to find the optimal number of trees, you can use **early stopping**. A simple way to implement this is to use the `staged_predict()` method: it returns an iterator over the predictions made by the ensemble at each stage of training (with one tree, two trees, etc.).

# Gradient Boosting with early stopping


```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=49)

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120, random_state=42)
gbrt.fit(X_train, y_train)

errors = [mean_squared_error(y_val, y_pred)
          for y_pred in gbrt.staged_predict(X_val)]
bst_n_estimators = np.argmin(errors) + 1

gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=bst_n_estimators, random_state=42)
gbrt_best.fit(X_train, y_train)

min_error = np.min(errors)
```


```python
plt.figure(figsize=(10, 4))

plt.subplot(121)
plt.plot(errors, 'b.-')
plt.plot([bst_n_estimators, bst_n_estimators], [0, min_error], 'k--')
plt.plot([0, 120], [min_error, min_error], 'k--')
plt.plot(bst_n_estimators, min_error, 'ko')
plt.text(bst_n_estimators, min_error*1.2, 'Minimum', ha='center', fontsize=14)
plt.axis([0, 120, 0, 0.01])
plt.xlabel("Number of trees")
plt.ylabel("Error", fontsize=16)
plt.title("Validation error", fontsize=14)

plt.subplot(122)
plot_predictions([gbrt_best], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
plt.title("Best model (%d trees)" % bst_n_estimators, fontsize=14)
plt.ylabel("$y$", fontsize=16, rotation=0)
plt.xlabel("$x_1$", fontsize=16)
```




    Text(0.5, 0, '$x_1$')




![png](https://pic.imgdb.cn/item/61715fca2ab3f51d91d81c57.png)


It is also possible to implement early stopping by actually stopping training early(instead of training a large number of trees first and then looking back to find the optimal number). You can do so by setting `warm_start=True`, which makes Scikit-Learn keep existing trees when the `fit()` method is called, allowing incremental training. The following code stops training when the validation error does not improve for five iterations in a row:


```python
gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True, random_state=42)

min_val_error = float("inf")
error_going_up = 0
for n_estimators in range(1, 120):
    gbrt.n_estimators = n_estimators
    gbrt.fit(X_train, y_train)
    y_pred = gbrt.predict(X_val)
    val_error = mean_squared_error(y_val, y_pred)
    if val_error < min_val_error:
        min_val_error = val_error
        error_going_up = 0
    else:
        error_going_up += 1
        if error_going_up == 5:
            break  # early stopping
```


```python
print(gbrt.n_estimators)
print("Minimum validation MSE:", min_val_error)
```

    61
    Minimum validation MSE: 0.002712853325235463


![53](https://pic.imgdb.cn/item/617159822ab3f51d91d14a95.png)


```python
import xgboost

xgb_reg = xgboost.XGBRFRegressor(random_state=42)
xgb_reg.fit(X_train, y_train)
y_pred = xgb_reg.predict(X_val)
mean_squared_error(y_val, y_pred)
```




    0.0029740468091323993




```python
xgb_reg = xgboost.XGBRFRegressor(random_state=42)

xgb_reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=2)
y_pred = xgb_reg.predict(X_val)
val_error = mean_squared_error(y_val, y_pred)  
print("Validation MSE:", val_error)           
```

    [0]	validation_0-rmse:0.05454
    Validation MSE: 0.0029740468091323993



```python
%timeit xgboost.XGBRFRegressor().fit(X_train, y_train)
```

    127 ms ± 4.32 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)



```python
%timeit GradientBoostingRegressor().fit(X_train, y_train)
```

    27.5 ms ± 2.1 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)


![54](https://pic.imgdb.cn/item/617159822ab3f51d91d14aa1.png)

# Dimensionality Reduction

## The Curse of  Dimensionality

![55](https://pic.imgdb.cn/item/617159822ab3f51d91d14aac.jpg)


## Projection and Manifold Learning


```python
from sklearn.datasets import make_swiss_roll
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
```


```python
from mpl_toolkits.mplot3d import Axes3D

axes = [-11.5, 14, -2, 23, -12, 15]

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=plt.cm.hot)
ax.view_init(10, -70)
ax.set_xlabel("$x_1$", fontsize=18)
ax.set_ylabel("$x_2$", fontsize=18)
ax.set_zlabel("$x_3$", fontsize=18)
ax.set_xlim(axes[0:2])
ax.set_ylim(axes[2:4])
ax.set_zlim(axes[4:6])
```




    (-12.0, 15.0)




![png](https://pic.imgdb.cn/item/61715fca2ab3f51d91d81c5c.png)



```python
plt.figure(figsize=(11, 4))

plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], c=t, cmap=plt.cm.hot)
plt.axis(axes[:4])
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$x_2$", fontsize=18, rotation=0)
plt.grid(True)

plt.subplot(122)
plt.scatter(t, X[:, 1], c=t, cmap=plt.cm.hot)
plt.axis([4, 15, axes[2], axes[3]])
plt.xlabel("$z_1$", fontsize=18)
plt.grid(True)

```


![png](https://pic.imgdb.cn/item/61715fca2ab3f51d91d81c65.png)



```python
from matplotlib import gridspec

axes = [-11.5, 14, -2, 23, -12, 15]

x2s = np.linspace(axes[2], axes[3], 10)
x3s = np.linspace(axes[4], axes[5], 10)
x2, x3 = np.meshgrid(x2s, x3s)

fig = plt.figure(figsize=(6, 5))
ax = plt.subplot(111, projection='3d')

positive_class = X[:, 0] > 5
X_pos = X[positive_class]
X_neg = X[~positive_class]
ax.view_init(10, -70)
ax.plot(X_neg[:, 0], X_neg[:, 1], X_neg[:, 2], "y^")
ax.plot_wireframe(5, x2, x3, alpha=0.5)
ax.plot(X_pos[:, 0], X_pos[:, 1], X_pos[:, 2], "gs")
ax.set_xlabel("$x_1$", fontsize=18)
ax.set_ylabel("$x_2$", fontsize=18)
ax.set_zlabel("$x_3$", fontsize=18)
ax.set_xlim(axes[0:2])
ax.set_ylim(axes[2:4])
ax.set_zlim(axes[4:6])


fig = plt.figure(figsize=(5, 4))
ax = plt.subplot(111)

plt.plot(t[positive_class], X[positive_class, 1], "gs")
plt.plot(t[~positive_class], X[~positive_class, 1], "y^")
plt.axis([4, 15, axes[2], axes[3]])
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18, rotation=0)
plt.grid(True)


fig = plt.figure(figsize=(6, 5))
ax = plt.subplot(111, projection='3d')

positive_class = 2 * (t[:] - 4) > X[:, 1]
X_pos = X[positive_class]
X_neg = X[~positive_class]
ax.view_init(10, -70)
ax.plot(X_neg[:, 0], X_neg[:, 1], X_neg[:, 2], "y^")
ax.plot(X_pos[:, 0], X_pos[:, 1], X_pos[:, 2], "gs")
ax.set_xlabel("$x_1$", fontsize=18)
ax.set_ylabel("$x_2$", fontsize=18)
ax.set_zlabel("$x_3$", fontsize=18)
ax.set_xlim(axes[0:2])
ax.set_ylim(axes[2:4])
ax.set_zlim(axes[4:6])


fig = plt.figure(figsize=(5, 4))
ax = plt.subplot(111)

plt.plot(t[positive_class], X[positive_class, 1], "gs")
plt.plot(t[~positive_class], X[~positive_class, 1], "y^")
plt.plot([4, 15], [0, 22], "b-", linewidth=2)
plt.axis([4, 15, axes[2], axes[3]])
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18, rotation=0)
plt.grid(True)
```


![png](https://pic.imgdb.cn/item/61715fca2ab3f51d91d81c70.png)



![png](https://pic.imgdb.cn/item/6171601a2ab3f51d91d878fb.png)



![png](https://pic.imgdb.cn/item/6171601a2ab3f51d91d87907.png)



![png](https://pic.imgdb.cn/item/6171601a2ab3f51d91d87914.png)


# PCA

![56](https://pic.imgdb.cn/item/617159c62ab3f51d91d1d973.png)


```python
# build 3D data
np.random.seed(4)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)
```

## PCA using SVD decompostion


```python
X_centered = X - X.mean(axis=0)

U, s, Vt = np.linalg.svd(X_centered)

c1 = Vt.T[:, 0]
c2 = Vt.T[:, 1]
W = Vt.T[:, :2]

X2D_using_svd = X_centered.dot(W)
```

## PCA with Scikit-Learn


```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X2D = pca.fit_transform(X) # mean centering is taken by sklearn atuomatically
```


```python
X2D[:5], X2D_using_svd[:5]
```




    (array([[ 1.26203346,  0.42067648],
            [-0.08001485, -0.35272239],
            [ 1.17545763,  0.36085729],
            [ 0.89305601, -0.30862856],
            [ 0.73016287, -0.25404049]]),
     array([[-1.26203346, -0.42067648],
            [ 0.08001485,  0.35272239],
            [-1.17545763, -0.36085729],
            [-0.89305601,  0.30862856],
            [-0.73016287,  0.25404049]]))



Notice that running PCA multiple times on slightly different datasets may result in different results. In general the only difference is that some axes may be flipped. In this example, PCA using Scikit-Learn gives the same projection as the one given by the SVD approach, except both axes are flipped:


```python
np.allclose(X2D, -X2D_using_svd)
```




    True



Recover the 3D points projected on the plane (PCA 2D subspace).

Of course, there was some loss of information during the projection step, so the recovered 3D points are not exactly equal to the original 3D points:


```python
X3D_inv = pca.inverse_transform(X2D)

np.allclose(X3D_inv, X)
```




    False



The inverse transform in the SVD approach looks like this:


```python
X3D_inv_using_svd = X2D_using_svd.dot(Vt[:2, :])
```

The reconstructions from both methods are not identical because Scikit-Learn's PCA class automatically takes care of reversing the mean centering, but if we subtract the mean, we get the same reconstruction:


```python
np.allclose(X3D_inv_using_svd, X3D_inv - pca.mean_)
```




    True




```python
pca.components_
```




    array([[-0.93636116, -0.29854881, -0.18465208],
           [ 0.34027485, -0.90119108, -0.2684542 ]])




```python
pca.explained_variance_, pca.explained_variance_ratio_
```




    (array([0.77830975, 0.1351726 ]), array([0.84248607, 0.14631839]))



## Choosing the right number of dimensions

you can set `n_components` to be a float between 0.0 and 1.0, indicating the ratio of variance you wish to preserve:


```python
pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum > 0.95) + 1
d
```




    2




```python
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X)
X_reduced.shape
```




    (60, 2)



## MNIST compression


```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.target = mnist.target.astype(np.uint8)
```


```python
from sklearn.model_selection import train_test_split

X = mnist["data"]
y = mnist["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y)
```


```python
pca = PCA()
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
d
```




    154




```python
plt.figure(figsize=(6,4))
plt.plot(cumsum, linewidth=3)
plt.axis([0, 400, 0, 1])
plt.xlabel("Dimensions")
plt.ylabel("Explained Variance")
plt.plot([d, d], [0, 0.95], "k:")
plt.plot([0, d], [0.95, 0.95], "k:")
plt.plot(d, 0.95, "ko")
plt.annotate("Elbow", xy=(65, 0.85), xytext=(70, 0.7),
             arrowprops=dict(arrowstyle="->"), fontsize=16)
plt.grid(True)
```


![png](https://pic.imgdb.cn/item/6171601a2ab3f51d91d8792b.png)



```python
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_train)

```


```python
X_recovered = pca.inverse_transform(X_reduced)
```


```python
def plot_digits(instances, images_per_row=5, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = plt.cm.binary, **options)
    plt.axis("off")
```


```python
plt.figure(figsize=(7, 4))
plt.subplot(121)
plot_digits(X_train[::2100])
plt.title("Original", fontsize=16)
plt.subplot(122)
plot_digits(X_recovered[::2100])
plt.title("Compressed", fontsize=16)
```




    Text(0.5, 1.0, 'Compressed')




![png](https://pic.imgdb.cn/item/6171601a2ab3f51d91d8793e.png)


## Randomized PCA


```python
rnd_pca = PCA(n_components=154, svd_solver='randomized')
X_reduced = rnd_pca.fit_transform(X_train)
```

## Incremental PCA


```python
from sklearn.decomposition import IncrementalPCA

n_batches = 100
inc_pca = IncrementalPCA(n_components=154)

for X_batch in np.array_split(X_train, n_batches):
    print(".", end='')
    inc_pca.partial_fit(X_batch)

X_reduced = inc_pca.transform(X_train)
```

### Using memmap()

Let's create the `memmap()` structure and copy the MNIST data into it. This would typically be done by a first program:


```python
filename = 'my_mnist.data'
m, n = X_train.shape

X_mm = np.memmap(filename, dtype='float32', mode='write', shape=(m, n))
X_mm[:] = X_train
```

Now deleting the `memmap()` object will trigger its Python finalizer, which ensures that the data is saved to disk.


```python
del X_mm
```

Next, another program would load the data and use it for training:


```python
X_mm = np.memmap(filename, dtype='float32', mode='readonly', shape=(m, n))

batch_size = m // n_batches

inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
inc_pca.fit(X_mm)
```




    IncrementalPCA(batch_size=525, n_components=154)



## Time Complexity

Let's time regular PCA against Incremental PCA and Randomized PCA, for various number of principal components:


```python
import time

for n_components in (2, 10, 154):
    print("n_components =", n_components)
    regular_pca = PCA(n_components=n_components, svd_solver="full")
    inc_pca = IncrementalPCA(n_components=n_components, batch_size=500)
    rnd_pca = PCA(n_components=n_components, random_state=42, svd_solver="randomized")

    for name, pca in (("PCA", regular_pca), ("Inc PCA", inc_pca), ("Rnd PCA", rnd_pca)):
        t1 = time.time()
        pca.fit(X_train)
        t2 = time.time()
        print("    {}: {:.1f} seconds".format(name, t2 - t1))
```

    n_components = 2
        PCA: 8.2 seconds
        Inc PCA: 36.3 seconds
        Rnd PCA: 1.3 seconds
    n_components = 10
        PCA: 7.4 seconds
        Inc PCA: 34.0 seconds
        Rnd PCA: 1.5 seconds
    n_components = 154
        PCA: 7.0 seconds
        Inc PCA: 45.1 seconds
        Rnd PCA: 5.0 seconds


Now let's compare PCA and Randomized PCA for datasets of different sizes (number of instances):


```python
times_rpca = []
times_pca = []
sizes = [1000, 10000, 20000, 30000, 40000, 50000, 70000, 100000, 200000, 500000]
for n_samples in sizes:
    X = np.random.randn(n_samples, 5)
    pca = PCA(n_components=2, svd_solver="randomized", random_state=42)
    t1 = time.time()
    pca.fit(X)
    t2 = time.time()
    times_rpca.append(t2 - t1)
    pca = PCA(n_components=2, svd_solver="full")
    t1 = time.time()
    pca.fit(X)
    t2 = time.time()
    times_pca.append(t2 - t1)

plt.plot(sizes, times_rpca, "b-o", label="RPCA")
plt.plot(sizes, times_pca, "r-s", label="PCA")
plt.xlabel("n_samples")
plt.ylabel("Training time")
plt.legend(loc="upper left")
plt.title("PCA and Randomized PCA time complexity ")
```




    Text(0.5, 1.0, 'PCA and Randomized PCA time complexity ')




![png](https://pic.imgdb.cn/item/617160682ab3f51d91d8e4f2.png)


And now let's compare their performance on datasets of 2,000 instances with various numbers of features:


```python
times_rpca = []
times_pca = []
sizes = [1000, 2000, 3000, 4000, 5000, 6000]
for n_features in sizes:
    X = np.random.randn(2000, n_features)
    pca = PCA(n_components=2, random_state=42, svd_solver="randomized")
    t1 = time.time()
    pca.fit(X)
    t2 = time.time()
    times_rpca.append(t2 - t1)
    pca = PCA(n_components=2, svd_solver="full")
    t1 = time.time()
    pca.fit(X)
    t2 = time.time()
    times_pca.append(t2 - t1)

plt.plot(sizes, times_rpca, "b-o", label="RPCA")
plt.plot(sizes, times_pca, "r-s", label="PCA")
plt.xlabel("n_features")
plt.ylabel("Training time")
plt.legend(loc="upper left")
plt.title("PCA and Randomized PCA time complexity ")
```




    Text(0.5, 1.0, 'PCA and Randomized PCA time complexity ')




![png](https://pic.imgdb.cn/item/617160682ab3f51d91d8e4fd.png)


## Kernel PCA


```python
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
```


```python
from sklearn.decomposition import KernelPCA

rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.04)
X_reduced = rbf_pca.fit_transform(X)
```


```python
from sklearn.decomposition import KernelPCA

lin_pca = KernelPCA(n_components = 2, kernel="linear", fit_inverse_transform=True)
rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)
sig_pca = KernelPCA(n_components = 2, kernel="sigmoid", gamma=0.001, coef0=1, fit_inverse_transform=True)

y = t > 6.9

plt.figure(figsize=(11, 4))
for subplot, pca, title in ((131, lin_pca, "Linear kernel"), (132, rbf_pca, "RBF kernel, $\gamma=0.04$"), (133, sig_pca, "Sigmoid kernel, $\gamma=10^{-3}, r=1$")):
    X_reduced = pca.fit_transform(X)
    # if subplot == 132:
    #     X_reduced_rbf = X_reduced
    
    plt.subplot(subplot)
    #plt.plot(X_reduced[y, 0], X_reduced[y, 1], "gs")
    #plt.plot(X_reduced[~y, 0], X_reduced[~y, 1], "y^")
    plt.title(title, fontsize=14)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
    plt.xlabel("$z_1$", fontsize=18)
    if subplot == 131:
        plt.ylabel("$z_2$", fontsize=18, rotation=0)
    plt.grid(True)
```


![png](https://pic.imgdb.cn/item/617160682ab3f51d91d8e508.png)


### Selecting a Kernel and Tuning Hyperparameters

![57](https://pic.imgdb.cn/item/617159c62ab3f51d91d1d97d.png)


```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

clf = Pipeline([
                ('kpca', KernelPCA(n_components=2)),
                ('log_reg', LogisticRegression(solver='lbfgs'))
])


param_grid = [{'kpca__gamma': np.linspace(0.03, 0.05, 10), 
               'kpca__kernel': ['rbf', 'sigmoid']}
              ]


grid_search = GridSearchCV(clf, param_grid, cv=3)

grid_search.fit(X, y)
```




    GridSearchCV(cv=3,
                 estimator=Pipeline(steps=[('kpca', KernelPCA(n_components=2)),
                                           ('log_reg', LogisticRegression())]),
                 param_grid=[{'kpca__gamma': array([0.03      , 0.03222222, 0.03444444, 0.03666667, 0.03888889,
           0.04111111, 0.04333333, 0.04555556, 0.04777778, 0.05      ]),
                              'kpca__kernel': ['rbf', 'sigmoid']}])




```python
print(grid_search.best_params_)
```

    {'kpca__gamma': 0.043333333333333335, 'kpca__kernel': 'rbf'}


Another approach, this time entirely unsupervised, is to select the kernel and hyperparameters that yield the lowest reconstruction error.


```python
rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433,
                    fit_inverse_transform=True)
X_reduced = rbf_pca.fit_transform(X)
X_preimage = rbf_pca.inverse_transform(X_reduced)
```


```python
from sklearn.metrics import mean_squared_error

mean_squared_error(X, X_preimage)
```




    7.779605696517346e-27



# Local Linear Embedding

![58](https://pic.imgdb.cn/item/617159c62ab3f51d91d1d986.png)


```python
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=41)
```


```python
from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)

X_reduced = lle.fit_transform(X)
```


```python
plt.title("Unrolled swiss roll using LLE", fontsize=14)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
plt.xlabel("$z_1$", fontsize=18)
plt.ylabel("$z_2$", fontsize=18)
plt.axis([-0.065, 0.055, -0.1, 0.12])
plt.grid(True)
```


![png](https://pic.imgdb.cn/item/617160682ab3f51d91d8e511.png)


# Other Dimensionality Reduction Techniques

![59](https://pic.imgdb.cn/item/617159c62ab3f51d91d1d98e.png)


```python
from sklearn.manifold import MDS

mds = MDS(n_components=2, random_state=42)
X_reduced_mds = mds.fit_transform(X)
```


```python
from sklearn.manifold import Isomap

isomap = Isomap(n_components=2)
X_reduced_isomap = isomap.fit_transform(X)
```


```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
X_reduced_tsne = tsne.fit_transform(X)
```


```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=2)
X_mnist = mnist["data"]
y_mnist = mnist["target"]
lda.fit(X_mnist, y_mnist)
X_reduced_lda = lda.transform(X_mnist)
```


```python
titles = ["MDS", "Isomap", "t-SNE"]

plt.figure(figsize=(11,4))

for subplot, title, X_reduced in zip((131, 132, 133), titles,
                                     (X_reduced_mds, X_reduced_isomap, X_reduced_tsne)):
    plt.subplot(subplot)
    plt.title(title, fontsize=14)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
    plt.xlabel("$z_1$", fontsize=18)
    if subplot == 131:
        plt.ylabel("$z_2$", fontsize=18, rotation=0)
    plt.grid(True)
```


![png](https://pic.imgdb.cn/item/617160682ab3f51d91d8e51d.png)


## Play with MNIST


```python
np.random.seed(42)

m = 2000
idx = np.random.permutation(70000)[:m]

X = mnist['data'][idx]
y = mnist['target'][idx]
```


```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
X_reduced = tsne.fit_transform(X)
```


```python
plt.figure(figsize=(13,10))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap="jet")
plt.axis('off')
plt.colorbar()
```




    <matplotlib.colorbar.Colorbar at 0x1f77482a908>




![png](https://pic.imgdb.cn/item/617160ce2ab3f51d91d96dbf.png)


Let's `create a plot_digits()` function that will draw a scatterplot (similar to the above scatterplots) plus write colored digits, with a minimum distance guaranteed between these digits. If the digit images are provided, they are plotted instead. This implementation was inspired from one of Scikit-Learn's excellent examples ([plot_lle_digits](https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html), based on a different digit dataset).


```python
from sklearn.preprocessing import MinMaxScaler
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

def plot_digits(X, y, min_distance=0.05, images=None, figsize=(13, 10)):
    # Let's scale the input features so that they range from 0 to 1
    X_normalized = MinMaxScaler().fit_transform(X)
    # Now we create the list of coordinates of the digits plotted so far.
    # We pretend that one is already plotted far away at the start, to
    # avoid `if` statements in the loop below
    neighbors = np.array([[10., 10.]])
    # The rest should be self-explanatory
    plt.figure(figsize=figsize)
    cmap = plt.cm.get_cmap("jet")
    digits = np.unique(y)
    for digit in digits:
        plt.scatter(X_normalized[y == digit, 0], X_normalized[y == digit, 1], c=[cmap(digit / 9)])
    plt.axis("off")
    ax = plt.gcf().gca()  # get current axes in current figure
    for index, image_coord in enumerate(X_normalized):
        closest_distance = np.linalg.norm(neighbors - image_coord, axis=1).min()
        if closest_distance > min_distance:
            neighbors = np.r_[neighbors, [image_coord]]
            if images is None:
                plt.text(image_coord[0], image_coord[1], str(int(y[index])),
                         color=cmap(y[index] / 9), fontdict={"weight": "bold", "size": 16})
            else:
                image = images[index].reshape(28, 28)
                imagebox = AnnotationBbox(OffsetImage(image, cmap="binary"), image_coord)
                ax.add_artist(imagebox)
```


```python
plot_digits(X_reduced, y)
```


![png](https://pic.imgdb.cn/item/617160ce2ab3f51d91d96dcb.png)



```python
plot_digits(X_reduced, y, images=X, figsize=(35, 25))
```


![png](https://pic.imgdb.cn/item/617160ce2ab3f51d91d96dd3.png)



```python
pca_tsne = Pipeline([
    ("pca", PCA(n_components=0.95, random_state=42)),
    ("tsne", TSNE(n_components=2, random_state=42)),
])
t0 = time.time()
X_pca_tsne_reduced = pca_tsne.fit_transform(X)
t1 = time.time()
print("PCA+t-SNE took {:.1f}s.".format(t1 - t0))
plot_digits(X_pca_tsne_reduced, y)
plt.show()
```

    PCA+t-SNE took 15.4s.



![png](https://pic.imgdb.cn/item/617160ce2ab3f51d91d96dde.png)


# Unsupervised Learning

![60](https://pic.imgdb.cn/item/617159c62ab3f51d91d1d9a0.png)

## Clustering

![61](https://pic.imgdb.cn/item/61715a152ab3f51d91d2ae24.png)

## K-Means


```python
from sklearn.datasets import make_blobs
```


```python
blob_centers = np.array(
    [[ 0.2,  2.3],
     [-1.5 ,  2.3],
     [-2.8,  1.8],
     [-2.8,  2.8],
     [-2.8,  1.3]])
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])

X, y = make_blobs(n_samples=2000, centers=blob_centers,
                  cluster_std=blob_std, random_state=7)
```


```python
def plot_clusters(X, y=None):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)
```


```python
plt.figure(figsize=(8, 4))
plot_clusters(X)
```


![png](https://pic.imgdb.cn/item/617160ce2ab3f51d91d96dec.png)



```python
from sklearn.cluster import KMeans

k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(X)
y_pred
```




    array([2, 4, 1, ..., 0, 1, 4])




```python
y_pred is kmeans.labels_
```




    True




```python
kmeans.cluster_centers_
```




    array([[-2.79290307,  2.79641063],
           [ 0.20876306,  2.25551336],
           [-2.80037642,  1.30082566],
           [-1.46679593,  2.28585348],
           [-2.80389616,  1.80117999]])




```python
X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
kmeans.predict(X_new)
```




    array([1, 1, 0, 0])




```python
# Decision Boundaries

def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)


def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=35, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=2, linewidths=12,
                color=cross_color, zorder=11, alpha=1)

def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)
```


```python
plt.figure(figsize=(8, 4))
plot_decision_boundaries(kmeans, X)
```


![png](https://pic.imgdb.cn/item/617161022ab3f51d91d9ace6.png)


### Hard Clustering vs Soft Clustering

Rather than arbitrarily choosing the closest cluster for each instance, which is called hard clustering, it might be better measure the distance of each instance to all 5 centroids. This is what the `transform()` method does:


```python
kmeans.transform(X_new)
```




    array([[2.9042344 , 0.32995317, 2.88633901, 1.49439034, 2.81093633],
           [5.84739223, 2.80290755, 5.84236351, 4.4759332 , 5.80730058],
           [0.29040966, 3.29399768, 1.71086031, 1.69136631, 1.21475352],
           [0.36159148, 3.21806371, 1.21567622, 1.54808703, 0.72581411]])



### K-Means Algorithm

The K-Means algorithm is one of the fastest clustering algorithms, and also one of the simplest:

- First initialize $k$ centroids randomly: $k$ distinct instances are chosen randomly from the dataset and the centroids are placed at their locations.
- Repeat until convergence (i.e., until the centroids stop moving):
  - Assign each instance to the closest centroid.
  - Update the centroids to be the mean of the instances that are assigned to them.


![62](https://pic.imgdb.cn/item/61715a152ab3f51d91d2ae36.png)

The KMeans class applies an optimized algorithm by default. To get the original K-Means algorithm (for educational purposes only), you must set `init="random"`, `n_init=1` and `algorithm="full"`. These hyperparameters will be explained below.


```python
kmeans_iter1 = KMeans(n_clusters=5, init="random", n_init=1,
                     algorithm="full", max_iter=1, random_state=0)
kmeans_iter2 = KMeans(n_clusters=5, init="random", n_init=1,
                     algorithm="full", max_iter=2, random_state=0)
kmeans_iter3 = KMeans(n_clusters=5, init="random", n_init=1,
                     algorithm="full", max_iter=3, random_state=0)
kmeans_iter1.fit(X)
kmeans_iter2.fit(X)
kmeans_iter3.fit(X)
```




    KMeans(algorithm='full', init='random', max_iter=3, n_clusters=5, n_init=1,
           random_state=0)




```python
plt.figure(figsize=(10, 8))

plt.subplot(321)
plot_data(X)
plot_centroids(kmeans_iter1.cluster_centers_, circle_color='r', cross_color='w')
plt.ylabel("$x_2$", fontsize=14, rotation=0)
plt.tick_params(labelbottom=False)
plt.title("Update the centroids (initially randomly)", fontsize=14)

plt.subplot(322)
plot_decision_boundaries(kmeans_iter1, X, show_xlabels=False, show_ylabels=False)
plt.title("Label the instances", fontsize=14)

plt.subplot(323)
plot_data(X)
plot_centroids(kmeans_iter2.cluster_centers_, circle_color='r', cross_color='w')
plt.ylabel("$x_2$", fontsize=14, rotation=0)
plt.tick_params(labelbottom=False)

plt.subplot(324)
plot_decision_boundaries(kmeans_iter2, X, show_xlabels=False, show_ylabels=False)

plt.subplot(325)
plot_data(X)
plot_centroids(kmeans_iter3.cluster_centers_, circle_color='r', cross_color='w')
plt.ylabel("$x_2$", fontsize=14, rotation=0)
plt.tick_params(labelbottom=False)

plt.subplot(326)
plot_decision_boundaries(kmeans_iter3, X, show_ylabels=False)
```


![png](https://pic.imgdb.cn/item/617161022ab3f51d91d9acf3.png)


Although the algorithm is guaranteed to converge, it may not converge to the right solution (i.e., it may converge to a local optimum): whether it does or not depends on the centroid initialization.

![63](https://pic.imgdb.cn/item/61715a152ab3f51d91d2ae4d.png)


```python
good_init = np.array([[-3, 3], [-3, 2], [-3, 1], [-1, 2], [0, 2]])
kmeans = KMeans(n_clusters=5, init=good_init, n_init=1)
kmeans.fit(X)
```




    KMeans(init=array([[-3,  3],
           [-3,  2],
           [-3,  1],
           [-1,  2],
           [ 0,  2]]),
           n_clusters=5, n_init=1)




```python
plot_decision_boundaries(kmeans, X)
```


![png](https://pic.imgdb.cn/item/617161022ab3f51d91d9acfe.png)



```python
kmeans.inertia_, kmeans.score(X)
```




    (211.59853725816828, -211.5985372581683)



![64](https://pic.imgdb.cn/item/61715a152ab3f51d91d2ae58.png)

To set the initialization to K-Means++, simply set `init="k-means++"` (this is actually the default):

![65](https://pic.imgdb.cn/item/61715a152ab3f51d91d2ae6d.png)

To use Elkan's variant of K-Means, just set `algorithm="elkan"`. Note that it does not support sparse data, so by default, Scikit-Learn uses "elkan" for dense data, and "full" (the regular K-Means algorithm) for sparse data.


```python
%timeit -n 50 KMeans(algorithm="elkan", random_state=42).fit(X)
```

    253 ms ± 6.12 ms per loop (mean ± std. dev. of 7 runs, 50 loops each)



```python
%timeit -n 50 KMeans(algorithm="full", random_state=42).fit(X)
```

    1.07 s ± 64.9 ms per loop (mean ± std. dev. of 7 runs, 50 loops each)


![66](https://pic.imgdb.cn/item/61715a582ab3f51d91d2ed4a.png)


```python
from sklearn.cluster import MiniBatchKMeans
minibatch_kmeans = MiniBatchKMeans(n_clusters=5, random_state=42)
minibatch_kmeans.fit(X)
minibatch_kmeans.inertia_
```

    D:\miniconda3\lib\site-packages\sklearn\cluster\_kmeans.py:888: UserWarning: MiniBatchKMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can prevent it by setting batch_size >= 2048 or by setting the environment variable OMP_NUM_THREADS=1
      f"MiniBatchKMeans is known to have a memory leak on "





    211.93186531476786




```python
from timeit import timeit
```


```python
times = np.empty((100, 2))
inertias = np.empty((100, 2))
for k in range(1, 101):
    kmeans_ = KMeans(n_clusters=k, random_state=42)
    minibatch_kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
    print("\r{}/{}".format(k, 100), end="")
    times[k-1, 0] = timeit("kmeans_.fit(X)", number=10, globals=globals())
    times[k-1, 1]  = timeit("minibatch_kmeans.fit(X)", number=10, globals=globals())
    inertias[k-1, 0] = kmeans_.inertia_
    inertias[k-1, 1] = minibatch_kmeans.inertia_
```


```python
plt.figure(figsize=(10,4))

plt.subplot(121)
plt.plot(range(1, 101), inertias[:, 0], "r--", label="K-Means")
plt.plot(range(1, 101), inertias[:, 1], "b.-", label="Mini-batch K-Means")
plt.xlabel("$k$", fontsize=16)
plt.title("Inertia", fontsize=14)
plt.legend(fontsize=14)
plt.axis([1, 100, 0, 100])

plt.subplot(122)
plt.plot(range(1, 101), times[:, 0], "r--", label="K-Means")
plt.plot(range(1, 101), times[:, 1], "b.-", label="Mini-batch K-Means")
plt.xlabel("$k$", fontsize=16)
plt.title("Training time (seconds)", fontsize=14)
plt.axis([1, 100, 0, 6])
```

### Finding the optimal number of clusters

What if the number of clusters was set to a lower or greater value than 5?


```python
kmeans_k3 = KMeans(n_clusters=3, random_state=42)
kmeans_k8 = KMeans(n_clusters=8, random_state=42)
kmeans_k3.fit(X)
kmeans_k8.fit(X)
```




    KMeans(random_state=42)




```python
fig, axes = plt.subplots(ncols=2, figsize=(10, 4))
plt.sca(axes[0])
plot_decision_boundaries(kmeans_k3, X)
plt.sca(axes[1])
plot_decision_boundaries(kmeans_k8, X)
```


![png](https://pic.imgdb.cn/item/617161022ab3f51d91d9ad08.png)



```python
kmeans_k3.inertia_, kmeans_k8.inertia_
```




    (653.2167190021556, 119.11983416102888)



![67](https://pic.imgdb.cn/item/61715a582ab3f51d91d2ed4e.png)


```python
kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X)
                for k in range(1, 10)]
inertias = [model.inertia_ for model in kmeans_per_k]
```


```python
plt.figure(figsize=(8, 3.5))
plt.plot(range(1, 10), inertias, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Inertia", fontsize=14)
plt.annotate('Elbow',
             xy=(4, inertias[3]),
             xytext=(0.55, 0.55),
             textcoords='figure fraction',
             fontsize=16,
             arrowprops=dict(facecolor='black', shrink=0.1)
            )
plt.axis([1, 8.5, 0, 1300])
```




    (1.0, 8.5, 0.0, 1300.0)




![png](https://pic.imgdb.cn/item/617161022ab3f51d91d9ad15.png)


![68](https://pic.imgdb.cn/item/61715a582ab3f51d91d2ed54.png)


```python
 from sklearn.metrics import silhouette_score

 silhouette_score(X, kmeans.labels_)
```




    0.655517642572828




```python
from sklearn.metrics import silhouette_samples

silhouette_samples(X, kmeans.labels_)
```




    array([0.66694071, 0.73103018, 0.48656361, ..., 0.846275  , 0.70971862,
           0.62920907])




```python
silhouette_scores = [silhouette_score(X, model.labels_)
                     for model in kmeans_per_k[1:]]
```


```python
plt.figure(figsize=(8, 3))
plt.plot(range(2, 10), silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score", fontsize=14)
plt.axis([1.8, 8.5, 0.55, 0.7])
```




    (1.8, 8.5, 0.55, 0.7)




![png](https://pic.imgdb.cn/item/617161752ab3f51d91da2bdb.png)


![69](https://pic.imgdb.cn/item/61715a582ab3f51d91d2ed60.png)


```python
from sklearn.metrics import silhouette_samples
from matplotlib.ticker import FixedLocator, FixedFormatter

plt.figure(figsize=(11, 9))

for k in (3, 4, 5, 6):
    plt.subplot(2, 2, k - 2)
    
    y_pred = kmeans_per_k[k - 1].labels_
    silhouette_coefficients = silhouette_samples(X, y_pred)

    padding = len(X) // 30
    pos = padding
    ticks = []
    for i in range(k):
        coeffs = silhouette_coefficients[y_pred == i]
        coeffs.sort()

        color = plt.cm.Spectral(i / k)
        plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ticks.append(pos + len(coeffs) // 2)
        pos += len(coeffs) + padding

    plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
    plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
    if k in (3, 5):
        plt.ylabel("Cluster")
    
    if k in (5, 6):
        plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.xlabel("Silhouette Coefficient")
    else:
        plt.tick_params(labelbottom=False)

    plt.axvline(x=silhouette_scores[k - 2], color="red", linestyle="--")
    plt.title("$k={}$".format(k), fontsize=16)
```


![png](https://pic.imgdb.cn/item/617161752ab3f51d91da2be7.png)


![70](https://pic.imgdb.cn/item/61715a582ab3f51d91d2ed68.png)

## Limits of K-Means

![71](https://pic.imgdb.cn/item/61715a8c2ab3f51d91d31ed5.png)


```python
X1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
X2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)
X2 = X2 + [6, -8]
X = np.r_[X1, X2]
y = np.r_[y1, y2]
```


```python
plot_clusters(X)
```


![png](https://pic.imgdb.cn/item/617161752ab3f51d91da2bf1.png)



```python
kmeans_good = KMeans(n_clusters=3, init=np.array([[-1.5, 2.5], [0.5, 0], [4, 0]]), n_init=1, random_state=42)
kmeans_bad = KMeans(n_clusters=3, random_state=42)
kmeans_good.fit(X)
kmeans_bad.fit(X)
```




    KMeans(n_clusters=3, random_state=42)




```python
plt.figure(figsize=(10, 3.2))

plt.subplot(121)
plot_decision_boundaries(kmeans_good, X)
plt.title("Inertia = {:.1f}".format(kmeans_good.inertia_), fontsize=14)

plt.subplot(122)
plot_decision_boundaries(kmeans_bad, X, show_ylabels=False)
plt.title("Inertia = {:.1f}".format(kmeans_bad.inertia_), fontsize=14)
```




    Text(0.5, 1.0, 'Inertia = 2179.5')




![png](https://pic.imgdb.cn/item/617161752ab3f51d91da2bff.png)


![72](https://pic.imgdb.cn/item/61715a8c2ab3f51d91d31edb.png)

# Using Clustering for Image Segmentation

![73](https://pic.imgdb.cn/item/61715a8c2ab3f51d91d31edf.png)


```python
from matplotlib.image import imread  # or 'from imageio import imread
# for each pixel there is a 3D vector containing the intensities of red, green, and blue, each between 0.0 and 1.0 (or between 0 and 255, if you use imageio.imread()).
img_path = './figs/ladybug.png'

image = imread(img_path)

plt.imshow(image)
```




    <matplotlib.image.AxesImage at 0x1c660839488>




![png](https://pic.imgdb.cn/item/617161752ab3f51d91da2c0b.png)


![74](https://pic.imgdb.cn/item/61715a8c2ab3f51d91d31ee8.png)


```python
X = image.reshape(-1, 3)
kmeans = KMeans(n_clusters=8, random_state=42).fit(X)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape(image.shape)
```


```python
segmented_imgs = []
n_colors = (10, 8, 6, 4, 2)
for n_clusters in n_colors:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    segmented_imgs.append(segmented_img.reshape(image.shape))
```


```python
plt.figure(figsize=(10,5))
plt.subplots_adjust(wspace=0.05, hspace=0.1)

plt.subplot(231)
plt.imshow(image)
plt.title("Original image")
plt.axis('off')

for idx, n_clusters in enumerate(n_colors):
    plt.subplot(232 + idx)
    plt.imshow(segmented_imgs[idx])
    plt.title("{} colors".format(n_clusters))
    plt.axis('off')
```


![png](https://pic.imgdb.cn/item/617161a52ab3f51d91da5bc6.png)


![75](https://pic.imgdb.cn/item/61715a8c2ab3f51d91d31ef1.png)

# Using Clustering for Preprocessing

Let's tackle the digits dataset which is a simple MNIST-like dataset containing 1,797 grayscale 8×8 images representing digits 0 to 9.


```python
from sklearn.datasets import load_digits
```


```python
X_digits, y_digits = load_digits(return_X_y=True)
```


```python
plt.imshow(X_digits[0].reshape(8, 8))
```




    <matplotlib.image.AxesImage at 0x1c6671c0388>




![png](https://pic.imgdb.cn/item/617161a52ab3f51d91da5bca.png)



```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, random_state=42)
```


```python
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
log_reg.fit(X_train, y_train)
```




    LogisticRegression(max_iter=5000, multi_class='ovr', random_state=42)




```python

log_reg_score = log_reg.score(X_test, y_test)
log_reg_score
```




    0.9688888888888889



Okay, that's our baseline: 96.89% accuracy. Let's see if we can do better by using K-Means as a preprocessing step. We will create a pipeline that will first cluster the training set into 50 clusters and replace the images with their distances to the 50 clusters, then apply a logistic regression model:

![76](https://pic.imgdb.cn/item/61715ad92ab3f51d91d36344.png)


```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("kmeans", KMeans(n_clusters=50, random_state=42)),
    ("log_reg", LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)),
])
pipeline.fit(X_train, y_train)

pipeline_score = pipeline.score(X_test, y_test)
pipeline_score
```




    0.9777777777777777



We chose the number of clusters $k$ completely arbitrarily, we can surely do better. Since K-Means is just a preprocessing step in a classification pipeline, finding a good value for $k$ is much simpler than earlier: there's no need to perform silhouette analysis or minimize the inertia, the best value of $k$ is simply the one that results in the best classification performance.


```python
from sklearn.model_selection import GridSearchCV

param_grid = dict(kmeans__n_clusters=range(2, 100))
grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)
grid_clf.fit(X_train, y_train)

# Warning: This may take close to 20 minutes to run, or more depending on your hardware.
```


```python
grid_clf.best_params_
```


```python
grid_clf.score(X_test, y_test)
```

# Clustering for Semi-supervised Learning

Another use case for clustering is in semi-supervised learning, when we have plenty of unlabeled instances and very few labeled instances.

Let's look at the performance of a logistic regression model when we only have 50 labeled instances.


```python
n_labeled = 50
log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", random_state=42)
log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])
log_reg.score(X_test, y_test)
```




    0.8333333333333334



It's much less than earlier of course. Let's see how we can do better. First, let's cluster the training set into 50 clusters, then for each cluster let's find the image closest to the centroid. We will call these images the **representative images**:


```python
k = 50
kmeans = KMeans(n_clusters=k, random_state=42)
X_digits_dist = kmeans.fit_transform(X_train)
representative_digit_idx = np.argmin(X_digits_dist, axis=0)
X_representative_digits = X_train[representative_digit_idx]
```


```python
plt.figure(figsize=(8, 2))
for index, X_representative_digit in enumerate(X_representative_digits):
    plt.subplot(k // 10, 10, index + 1)
    plt.imshow(X_representative_digit.reshape(8, 8), cmap="binary", interpolation="bilinear")
    plt.axis('off')
```


![png](https://pic.imgdb.cn/item/617161a52ab3f51d91da5bd0.png)



```python
y_train[representative_digit_idx]
```




    array([4, 8, 0, 6, 8, 3, 7, 7, 9, 2, 5, 5, 8, 5, 2, 1, 2, 9, 6, 1, 1, 6,
           9, 0, 8, 3, 0, 7, 4, 1, 6, 5, 2, 4, 1, 8, 6, 3, 9, 2, 4, 2, 9, 4,
           7, 6, 2, 3, 1, 1])




```python
# label these representive images
y_representative_digits = np.array([
    4, 8, 0, 6, 8, 3, 7, 7, 9, 2, 
    5, 5, 8, 5, 2, 1, 2, 9, 6, 1, 
    1, 6, 9, 0, 8, 3, 0, 7, 4, 1, 
    6, 5, 2, 4, 1, 8, 6, 3, 9, 2, 
    4, 2, 9, 4, 7, 6, 2, 3, 1, 1])
```

Now we have a dataset with just 50 labeled instances, but instead of being completely random instances, each of them is a representative image of its cluster. Let's see if the performance is any better:


```python
log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
log_reg.fit(X_representative_digits, y_representative_digits)
log_reg.score(X_test, y_test)
```




    0.9222222222222223



Wow! We jumped from 83.3% accuracy to 92.2%, although we are still only training the model on 50 instances. Since it's often costly and painful to label instances, especially when it has to be done manually by experts, it's a good idea to make them label representative instances rather than just random instances.

But perhaps we can go one step further: what if we propagated the labels to all the other instances in the same cluster?


```python
y_train_propagated = np.empty(len(X_train), dtype=np.int32)
for i in range(k):
    y_train_propagated[kmeans.labels_==i] = y_representative_digits[i]
```


```python
log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
log_reg.fit(X_train, y_train_propagated)
log_reg.score(X_test, y_test)
```




    0.9333333333333333



We got a tiny little accuracy boost. Better than nothing, but we should probably have propagated the labels only to the instances closest to the centroid, because by propagating to the full cluster, we have certainly included some outliers. Let's only propagate the labels to the 75th percentile closest to the centroid:


```python
percentile_closest = 75

X_cluster_dist = X_digits_dist[np.arange(len(X_train)), kmeans.labels_]
for i in range(k):
    in_cluster = (kmeans.labels_ == i)
    cluster_dist = X_cluster_dist[in_cluster]
    cutoff_distance = np.percentile(cluster_dist, percentile_closest)
    above_cutoff = (X_cluster_dist > cutoff_distance)
    X_cluster_dist[in_cluster & above_cutoff] = -1

# 将每个类中距离centroid超过75分位数的点标记为-1
```


```python
partially_propagated = (X_cluster_dist != -1)
X_train_partially_propagated = X_train[partially_propagated]
y_train_partially_propagated = y_train_propagated[partially_propagated]
```


```python
log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
log_reg.fit(X_train_partially_propagated, y_train_partially_propagated)
log_reg.score(X_test, y_test)
```




    0.9355555555555556



A bit better. With just 50 labeled instances (just 5 examples per class on average!), we got 93.6% performance, which is getting closer to the performance of logistic regression on the fully labeled digits dataset (which was 96.9%).

This is because the propagated labels are actually pretty good: their accuracy is about 97.5%:


```python
np.mean(y_train_partially_propagated == y_train[partially_propagated])
```




    0.9750747756729811



![77](https://pic.imgdb.cn/item/61715ad92ab3f51d91d36349.png)

# DBSCAN

![78](https://pic.imgdb.cn/item/61715ad92ab3f51d91d3634f.png)


```python
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)
```


```python
from sklearn.cluster import DBSCAN
```


```python
dbscan = DBSCAN(eps=0.05, min_samples=5)
dbscan.fit(X)
```




    DBSCAN(eps=0.05)




```python
dbscan.labels_[:10]
```




    array([ 0,  2, -1, -1,  1,  0,  0,  0,  2,  5], dtype=int64)




```python
dbscan.core_sample_indices_[:10]
```




    array([ 0,  4,  5,  6,  7,  8, 10, 11, 12, 13], dtype=int64)




```python
dbscan.components_[:3]
```




    array([[-0.02137124,  0.40618608],
           [-0.84192557,  0.53058695],
           [ 0.58930337, -0.32137599]])




```python
dbscan2 = DBSCAN(eps=0.2)
dbscan2.fit(X)
```




    DBSCAN(eps=0.2)




```python
def plot_dbscan(dbscan, X, size, show_xlabels=True, show_ylabels=True):
    core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_mask[dbscan.core_sample_indices_] = True
    anomalies_mask = dbscan.labels_ == -1
    non_core_mask = ~(core_mask | anomalies_mask)

    cores = dbscan.components_
    anomalies = X[anomalies_mask]
    non_cores = X[non_core_mask]
    
    plt.scatter(cores[:, 0], cores[:, 1],
                c=dbscan.labels_[core_mask], marker='o', s=size, cmap="Paired")
    plt.scatter(cores[:, 0], cores[:, 1], marker='*', s=20, c=dbscan.labels_[core_mask])
    plt.scatter(anomalies[:, 0], anomalies[:, 1],
                c="r", marker="x", s=100)
    plt.scatter(non_cores[:, 0], non_cores[:, 1], c=dbscan.labels_[non_core_mask], marker=".")
    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)
    plt.title("eps={:.2f}, min_samples={}".format(dbscan.eps, dbscan.min_samples), fontsize=14)
```


```python
plt.figure(figsize=(9, 3.2))

plt.subplot(121)
plot_dbscan(dbscan, X, size=100)

plt.subplot(122)
plot_dbscan(dbscan2, X, size=600, show_ylabels=False)
```


![png](https://pic.imgdb.cn/item/617161a62ab3f51d91da5bd7.png)


![79](https://pic.imgdb.cn/item/61715ad92ab3f51d91d36359.png)


```python
dbscan2.core_sample_indices_.shape
```




    (1000,)




```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(dbscan2.components_, dbscan2.labels_[dbscan2.core_sample_indices_])
```




    KNeighborsClassifier(n_neighbors=50)




```python
X_new = np.array([[-0.5, 0], [0, 0.5], [1, -0.1], [2, 1]])
knn.predict(X_new)
```




    array([1, 0, 1, 0], dtype=int64)




```python
knn.predict_proba(X_new)
```




    array([[0.18, 0.82],
           [1.  , 0.  ],
           [0.12, 0.88],
           [1.  , 0.  ]])




```python
plt.figure(figsize=(6, 3))
plot_decision_boundaries(knn, X, show_centroids=False)
plt.scatter(X_new[:, 0], X_new[:, 1], c="b", marker="+", s=200, zorder=10)
```




    <matplotlib.collections.PathCollection at 0x1c66986dc08>




![png](https://pic.imgdb.cn/item/617161a62ab3f51d91da5be5.png)


![80](https://pic.imgdb.cn/item/61715ad92ab3f51d91d36364.png)


```python
y_dist, y_pred_idx = knn.kneighbors(X_new, n_neighbors=1)
y_pred = dbscan2.labels_[dbscan2.core_sample_indices_][y_pred_idx]
y_pred[y_dist > 0.2] = -1
y_pred.ravel()
```




    array([-1,  0,  1, -1], dtype=int64)



![81](https://pic.imgdb.cn/item/61715b232ab3f51d91d3ac2e.png)

# Gaussian Mixtures

![82](https://pic.imgdb.cn/item/61715b232ab3f51d91d3ac33.png)


```python
X1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
X2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)
X2 = X2 + [6, -8]
X = np.r_[X1, X2]
y = np.r_[y1, y2]
```


```python
from sklearn.mixture import GaussianMixture

gm = GaussianMixture(n_components=3, n_init=10, random_state=42)
gm.fit(X)
```




    GaussianMixture(n_components=3, n_init=10, random_state=42)




```python
gm.weights_
```




    array([0.39025715, 0.40007391, 0.20966893])




```python
gm.means_
```




    array([[ 0.05131611,  0.07521837],
           [-1.40763156,  1.42708225],
           [ 3.39893794,  1.05928897]])




```python
gm.covariances_
```




    array([[[ 0.68799922,  0.79606357],
            [ 0.79606357,  1.21236106]],
    
           [[ 0.63479409,  0.72970799],
            [ 0.72970799,  1.1610351 ]],
    
           [[ 1.14833585, -0.03256179],
            [-0.03256179,  0.95490931]]])



![83](https://pic.imgdb.cn/item/61715b232ab3f51d91d3ac3c.png)


```python
gm.converged_, gm.n_iter_
```




    (True, 4)




```python
gm.predict(X)
```




    array([0, 0, 1, ..., 2, 2, 2], dtype=int64)




```python
gm.predict_proba(X)
```




    array([[9.76741808e-01, 6.78581203e-07, 2.32575136e-02],
           [9.82832955e-01, 6.76173663e-04, 1.64908714e-02],
           [7.46494398e-05, 9.99923327e-01, 2.02398402e-06],
           ...,
           [4.26050456e-07, 2.15512941e-26, 9.99999574e-01],
           [5.04987704e-16, 1.48083217e-41, 1.00000000e+00],
           [2.24602826e-15, 8.11457779e-41, 1.00000000e+00]])




```python
X_new, y_new = gm.sample(6)
X_new
```




    array([[-0.86944074, -0.32767626],
           [ 0.29836051,  0.28297011],
           [-2.8014927 , -0.09047309],
           [ 3.98203732,  1.49951491],
           [ 3.81677148,  0.53095244],
           [ 2.84104923, -0.73858639]])




```python
y_new
```




    array([0, 0, 1, 2, 2, 2])



You can also estimate the log of the probability density function (PDF) at any location using the `score_samples()` method:


```python
gm.score_samples(X)
```




    array([-2.60768954, -3.57110232, -3.32987086, ..., -3.51347241,
           -4.39798588, -3.80746532])



Let's check that the PDF integrates to 1 over the whole space. We just take a large square around the clusters, and chop it into a grid of tiny squares, then we compute the approximate probability that the instances will be generated in each tiny square (by multiplying the PDF at one corner of the tiny square by the area of the square), and finally summing all these probabilities). The result is very close to 1:


```python
resolution = 100
grid = np.arange(-10, 10, 1 / resolution)
xx, yy = np.meshgrid(grid, grid)
X_full = np.vstack([xx.ravel(), yy.ravel()]).T

pdf = np.exp(gm.score_samples(X_full))
pdf_probas = pdf * (1 / resolution) ** 2
pdf_probas.sum()
```




    0.9999999999215021




```python
from matplotlib.colors import LogNorm

def plot_gaussian_mixture(clusterer, X, resolution=1000, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = -clusterer.score_samples(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z,
                 norm=LogNorm(vmin=1.0, vmax=30.0),
                 levels=np.logspace(0, 2, 12))
    plt.contour(xx, yy, Z,
                norm=LogNorm(vmin=1.0, vmax=30.0),
                levels=np.logspace(0, 2, 12),
                linewidths=1, colors='k')

    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z,
                linewidths=2, colors='r', linestyles='dashed')
    
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
    plot_centroids(clusterer.means_, clusterer.weights_)

    plt.xlabel("$x_1$", fontsize=14)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)
```


```python
plt.figure(figsize=(8, 4))

plot_gaussian_mixture(gm, X)
```


![png](https://pic.imgdb.cn/item/617161ee2ab3f51d91daa76d.png)


![84](https://pic.imgdb.cn/item/61715b232ab3f51d91d3ac45.png)

You can impose constraints on the covariance matrices that the algorithm looks for by setting the covariance_type hyperparameter:

- "full" (default): no constraint, all clusters can take on any ellipsoidal shape of any size.
- "tied": all clusters must have the same shape, which can be any ellipsoid (i.e., they all share the same covariance matrix).
- "spherical": all clusters must be spherical, but they can have different diameters (i.e., different variances).
- "diag": clusters can take on any ellipsoidal shape of any size, but the ellipsoid's axes must be parallel to the axes (i.e., the covariance matrices must be diagonal).


```python
gm_full = GaussianMixture(n_components=3, n_init=10, covariance_type="full", random_state=42)
gm_tied = GaussianMixture(n_components=3, n_init=10, covariance_type="tied", random_state=42)
gm_spherical = GaussianMixture(n_components=3, n_init=10, covariance_type="spherical", random_state=42)
gm_diag = GaussianMixture(n_components=3, n_init=10, covariance_type="diag", random_state=42)
gm_full.fit(X)
gm_tied.fit(X)
gm_spherical.fit(X)
gm_diag.fit(X)
```




    GaussianMixture(covariance_type='diag', n_components=3, n_init=10,
                    random_state=42)




```python
def compare_gaussian_mixtures(gm1, gm2, X):
    plt.figure(figsize=(9, 4))

    plt.subplot(121)
    plot_gaussian_mixture(gm1, X)
    plt.title('covariance_type="{}"'.format(gm1.covariance_type), fontsize=14)

    plt.subplot(122)
    plot_gaussian_mixture(gm2, X, show_ylabels=False)
    plt.title('covariance_type="{}"'.format(gm2.covariance_type), fontsize=14)

compare_gaussian_mixtures(gm_tied, gm_spherical, X)
```


![png](https://pic.imgdb.cn/item/617161ee2ab3f51d91daa772.png)


![85](https://pic.imgdb.cn/item/61715b232ab3f51d91d3ac4f.png)

## Anomaly Detection Using Gaussian Mixtures

![86](https://pic.imgdb.cn/item/61715b902ab3f51d91d40b2e.png)


```python
densities = gm.score_samples(X)
density_threshold = np.percentile(densities, 4)
anomalies = X[densities < density_threshold]
```


```python
plt.figure(figsize=(8, 4))

plot_gaussian_mixture(gm, X)
plt.scatter(anomalies[:, 0], anomalies[:, 1], color='r', marker='*')
plt.ylim(top=5.1)
```




    (-2.948604775181289, 5.1)




![png](https://pic.imgdb.cn/item/617161ee2ab3f51d91daa778.png)


# Selecting the Number of Clusters

![87](https://pic.imgdb.cn/item/61715b902ab3f51d91d40b32.png)


```python
gm.aic(X), gm.bic(X)
```




    (8102.521720382148, 8189.747000497186)




```python
n_clusters = 3
n_dims = 2
n_params_for_weights = n_clusters - 1
n_params_for_means = n_clusters * n_dims
n_params_for_covariance = n_clusters * n_dims * (n_dims + 1) // 2
n_params = n_params_for_weights + n_params_for_means + n_params_for_covariance
max_log_likelihood = gm.score(X) * len(X) # log(L^)
bic = np.log(len(X)) * n_params - 2 * max_log_likelihood
aic = 2 * n_params - 2 * max_log_likelihood

aic, bic
```




    (8102.521720382148, 8189.747000497186)




```python
gms_per_k = [GaussianMixture(n_components=k, n_init=10, random_state=42).fit(X) for k in range(1, 11)]
```



```python
bics = [model.bic(X) for model in gms_per_k]
aics = [model.aic(X) for model in gms_per_k]
```


```python
plt.figure(figsize=(8, 3))
plt.plot(range(1, 11), bics, "bo-", label="BIC")
plt.plot(range(1, 11), aics, "go--", label="AIC")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Information Criterion", fontsize=14)
plt.axis([1, 9.5, np.min(aics) - 50, np.max(aics) + 50])
plt.annotate('Minimum',
             xy=(3, bics[2]),
             xytext=(0.35, 0.6),
             textcoords='figure fraction',
             fontsize=14,
             arrowprops=dict(facecolor='black', shrink=0.1)
            )
plt.legend()
```




    <matplotlib.legend.Legend at 0x1c66402f788>




![png](https://pic.imgdb.cn/item/617161ee2ab3f51d91daa782.png)



```python
min_bic = np.infty

for k in range(1, 11):
    for covariance_type in ("full", "tied", "spherical", "diag"):
        bic = GaussianMixture(n_components=k, n_init=10,
                              covariance_type=covariance_type,
                              random_state=42).fit(X).bic(X)
        if bic < min_bic:
            min_bic = bic
            best_k = k
            best_covariance_type = covariance_type
```


```python
best_k, best_covariance_type
```

## Bayesian Gaussian Mixture Models

![88](https://pic.imgdb.cn/item/61715b902ab3f51d91d40b3a.png)


```python
from sklearn.mixture import BayesianGaussianMixture
bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)
bgm.fit(X)
```




    BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)




```python
np.round(bgm.weights_, 2)
```




    array([0.4 , 0.21, 0.4 , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ])




```python
plt.figure(figsize=(8, 5))
plot_gaussian_mixture(bgm, X)
```


![png](https://pic.imgdb.cn/item/617161ee2ab3f51d91daa786.png)



```python
bgm_low = BayesianGaussianMixture(n_components=10, max_iter=1000, n_init=1,
                                  weight_concentration_prior=0.01, random_state=42)
bgm_high = BayesianGaussianMixture(n_components=10, max_iter=1000, n_init=1,
                                  weight_concentration_prior=10000, random_state=42)
nn = 73
bgm_low.fit(X[:nn])
bgm_high.fit(X[:nn])
```




    BayesianGaussianMixture(max_iter=1000, n_components=10, random_state=42,
                            weight_concentration_prior=10000)




```python
np.round(bgm_low.weights_, 2), np.round(bgm_high.weights_, 2)
```




    (array([0.52, 0.48, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ]),
     array([0.01, 0.18, 0.27, 0.11, 0.01, 0.01, 0.01, 0.01, 0.37, 0.01]))




```python
plt.figure(figsize=(9, 4))

plt.subplot(121)
plot_gaussian_mixture(bgm_low, X[:nn])
plt.title("weight_concentration_prior = 0.01", fontsize=14)

plt.subplot(122)
plot_gaussian_mixture(bgm_high, X[:nn], show_ylabels=False)
plt.title("weight_concentration_prior = 10000", fontsize=14)
```




    Text(0.5, 1.0, 'weight_concentration_prior = 10000')




![png](https://pic.imgdb.cn/item/617162302ab3f51d91daf29b.png)


![89](https://pic.imgdb.cn/item/61715b902ab3f51d91d40b42.png)

For example, the distributions and lower-bound equations used in Scikit-Learn’s `BayesianGaussianMixture` class are presented in the [documentation](https://scikit-learn.org/0.19/modules/dp-derivation.html).

![90](https://pic.imgdb.cn/item/61715b902ab3f51d91d40b4c.png)


```python
X_moons, y_moons = make_moons(n_samples=1000, noise=0.05, random_state=42)
```


```python
bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)
bgm.fit(X_moons)
```




    BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)




```python
plt.figure(figsize=(9, 3.2))

plt.subplot(121)
plot_data(X_moons)
plt.xlabel("$x_1$", fontsize=14)
plt.ylabel("$x_2$", fontsize=14, rotation=0)

plt.subplot(122)
plot_gaussian_mixture(bgm, X_moons, show_ylabels=False)
```


![png](https://pic.imgdb.cn/item/617162302ab3f51d91daf2a0.png)


![91](https://pic.imgdb.cn/item/61715bc42ab3f51d91d43ba7.png)

