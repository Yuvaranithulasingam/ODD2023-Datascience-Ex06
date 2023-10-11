# ODD2023-Datascience-Ex06

## AIM:
To read the given data and perform Feature Transformation process and save the data to a file.

## EXPLANATION:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

## ALGORITHM:
### STEP 1:
Read the given Data
### STEP 2:
Clean the Data Set using Data Cleaning Process
### STEP 3:
Apply Feature Transformation techniques to all the features of the data set
### STEP 4:
Print the transformed features

## PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
df = pd.read_csv("/content/Data_to_Transform.csv")
print(df)
df.head()
df.isnull().sum()
df.info()
df.describe()

df1 = df.copy()
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df1['Highly Positive Skew'] = np.log(df1['Highly Positive Skew'])
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

df2 = df.copy()
df2['Highly Positive Skew'] = 1/df2['Highly Positive Skew']
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df3 = df.copy()
df3['Highly Positive Skew'] = df3['Highly Positive Skew']**(1/1.2)
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df4 = df.copy()
df4['Moderate Positive Skew_1'],parameters =stats.yeojohnson(df4['Moderate Positive Skew'])
sm.qqplot(df4['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer 
trans = PowerTransformer("yeo-johnson")
df5 = df.copy()
df5['Moderate Negative Skew_1'] = pd.DataFrame(trans.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_1'],line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df5['Moderate Negative Skew_2'] = pd.DataFrame(qt.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_2'],line='45')
plt.show()
```
# OUTPUT:

### Original Data:
![image](https://github.com/Yuvaranithulasingam/ODD2023-Datascience-Ex06/assets/121418522/c06332e3-2093-4674-b862-ea5fd699a378)

### Data information:
![image](https://github.com/Yuvaranithulasingam/ODD2023-Datascience-Ex06/assets/121418522/22b7a6f3-8af3-4c8a-8a6e-57f43d9bd12f)

### Data describe:
![image](https://github.com/Yuvaranithulasingam/ODD2023-Datascience-Ex06/assets/121418522/33133b16-e895-419f-a38e-6c2616f93dd6)

### Before transformation:
![image](https://github.com/Yuvaranithulasingam/ODD2023-Datascience-Ex06/assets/121418522/8e207720-fa46-4a0d-8d54-9dcd99e32c95)

![image](https://github.com/Yuvaranithulasingam/ODD2023-Datascience-Ex06/assets/121418522/3dd44293-5295-4ca8-aea0-1cd36a011aab)

![image](https://github.com/Yuvaranithulasingam/ODD2023-Datascience-Ex06/assets/121418522/419d60ba-f137-4ee4-968c-b6e6479883ac)

![image](https://github.com/Yuvaranithulasingam/ODD2023-Datascience-Ex06/assets/121418522/a65c904d-1b29-4896-ad91-c95edc0ba0ce)

### Log transformation:
![image](https://github.com/Yuvaranithulasingam/ODD2023-Datascience-Ex06/assets/121418522/f5dd7244-b004-4311-b598-4a499c85969f)

### Reciprocal transformation:
![image](https://github.com/Yuvaranithulasingam/ODD2023-Datascience-Ex06/assets/121418522/a36cbcac-fa50-4836-b1af-c1fcc4ae89f2)

### Square root transformation:
![image](https://github.com/Yuvaranithulasingam/ODD2023-Datascience-Ex06/assets/121418522/e4e36eb0-26ea-49a4-9089-acc48da671c9)

![image](https://github.com/Yuvaranithulasingam/ODD2023-Datascience-Ex06/assets/121418522/5cdf222d-b13f-4b69-9749-9ef9427d7ccd)

### Power transformation:
![image](https://github.com/Yuvaranithulasingam/ODD2023-Datascience-Ex06/assets/121418522/1c78172d-9337-4f19-a89f-078c2320dd78)

![image](https://github.com/Yuvaranithulasingam/ODD2023-Datascience-Ex06/assets/121418522/b26bae30-af3d-46a4-9448-2dbaabd66564)

### Quantile transformation:
![image](https://github.com/Yuvaranithulasingam/ODD2023-Datascience-Ex06/assets/121418522/c571497c-130a-4055-9656-2b52c5695cf4)

## RESULT:
Thus feature transformation is done for the given dataset.
