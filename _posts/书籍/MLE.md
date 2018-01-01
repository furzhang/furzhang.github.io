
# Maximum Likehood Estimation (MLE)

This page introduces a basic method in statistics: Maximum Likehood Estimation (ELM). To understand what ELM is, we first need to figure out what is likehood. Statistics, regardless of frequentist or baysian, is all about fitting data to our defined model, which defines the likehood function. The goal of fitting is to find out the parameters of our model which fit our data best, and mathematically speaking, best fitting is defined as maximum likehood. 
For example, if we have a sequence of numbers X (like heights of a population), it would be reasonable to assume that this sequence follows normal distribution. The the likehood would be defined by the probability densitiy function (pdf) of the normal distibution. The the question is to find out parameters, in this case mean and standard deviation of normal distribution, that fit our data best. 
In the following section, we will explore how to find out the best fitting parameters for our model with MLE. For demostration purpose, we will use a simple problem defined in last paragraph.

## Step0: Enviroment Initializing


```python
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline
from scipy import stats
from statsmodels.base.model import GenericLikelihoodModel
```

## Step1: Load data

Here we will use dataset availiable at: http://people.ucsc.edu/~cdobkin/NHIS%202007%20data.csv
After downloading, we will import to python:


```python
DataSet = pd.read_csv('./NHIS 2007 data.csv')
DataSet.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HHX</th>
      <th>FMX</th>
      <th>FPX</th>
      <th>SEX</th>
      <th>BMI</th>
      <th>SLEEP</th>
      <th>educ</th>
      <th>height</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>33.36</td>
      <td>8</td>
      <td>16</td>
      <td>74</td>
      <td>260</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>26.54</td>
      <td>7</td>
      <td>14</td>
      <td>70</td>
      <td>185</td>
    </tr>
    <tr>
      <th>2</th>
      <td>69</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>32.13</td>
      <td>7</td>
      <td>9</td>
      <td>61</td>
      <td>170</td>
    </tr>
    <tr>
      <th>3</th>
      <td>87</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>26.62</td>
      <td>8</td>
      <td>14</td>
      <td>68</td>
      <td>175</td>
    </tr>
    <tr>
      <th>4</th>
      <td>88</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>27.13</td>
      <td>8</td>
      <td>13</td>
      <td>66</td>
      <td>168</td>
    </tr>
  </tbody>
</table>
</div>



This dataset contains 9 variables concerning a person's health infromation. In this example, we will explore weight only for demonstration purpose


```python
Weight = DataSet['weight']
Weight.head()
```




    0    260
    1    185
    2    170
    3    175
    4    168
    Name: weight, dtype: int64



Let's see how weight is distribution across this population (4784 people)


```python
hist_Weight = plt.hist(Weight)
```


![png](MLE_files/MLE_10_0.png)


As you can see, there is a group of people who have weights more that 800 pounds. According to our common sense this is impossible, so we think that these data points are outliers and should be ruled out.


```python
Weight = Weight.drop(Weight[Weight>800].index);
hist_Weight = plt.hist(Weight,20);
```


![png](MLE_files/MLE_12_0.png)


Now the data looks reasonable and we can fit a normal distribution to this dataset.

## Step2: Data Fitting

Here we will use GenericLikelihoodModel from statsmodels library to perform MLE. To do so ,we define a class that inherit from GenericLikelihoodModel: MLE_normal.


```python
class MLE_normal(GenericLikelihoodModel):
    def _init_(self, endog,exog=None,**kwds):
        if exog is None:
            exog = np.zeros_like(endog)
        super(MLE_normal,self)._init_(endog,exog,**kwds)
    
    def nloglikeobs(self,param):
        mu = param[0];
        std = param[1];
        ll = -np.log(stats.norm.pdf(self.endog,mu,std))
        return ll
    
    def fit(self,start_params=None,maxiter=10000,maxfun=5000,**kwds):
        if start_params is None:
            mu_start = np.mean(self.endog);
            std_start = np.std(self.endog);
            start_params = np.array([mu_start,std_start])
        return super(MLE_normal,self).fit(start_params=start_params,
                                         maxiter = maxiter,maxfun=maxfun,**kwds)
```

After defining the class, we are ready to fit the model on our data.

## Step3:Model Fitting


```python
Model = MLE_normal(Weight)
Results = Model.fit()
```

    Optimization terminated successfully.
             Current function value: 5.077668
             Iterations: 33
             Function evaluations: 67
    


```python
Params = Results.params #Extract the parameters [mu,std]
```

Now let's check the fitting results by superimpose the original distribution with the fitting curve


```python
x = np.linspace(50,350,1000)
Weight_hist = plt.hist(Weight,20,normed=True)
h = plt.plot(x,stats.norm.pdf(x,Params[0],Params[1]),lw=2)
plt.show()
```


![png](MLE_files/MLE_22_0.png)


As shown in this figure, this is a pretty reasonable fitting.

## Summary

In this example, we demonstrated how to use MLE to perform fitting with the weight example data. This process involved 3 steps: 1. Visiualize data to find a reseasonal model and rule out outliers. 2. Define a class inherited from GenericLikelihoododel by overwriting _init_ function, nliglikeobs function, and fit function to our customized model. 3. Do fitting and inspect the result 
