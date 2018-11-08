# 02-11-18

### 1. Carry out all the solutions from 31-10-18.

Finish the 6 procedures from 31-10-18 to completion on or before 10-11-18.  

### 2. Help Amit with his research.

The different affecting factors of stock market prices.  
The risk associated with each of these factors wrt some stock market price.  

# 31-10-18

### 1. Pseucocodic form of the algorithms.

### 2. Existing mathematical literature for each model.

Find and document each of the used models.

Code not required, but pseucodic algorithmic understanding and documentation is important.  

------------------------------------------------------------------------------------------------------------------------------

For long-term SMP, linear_model from scikit-learn has been used:  
linear regression: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression  
ridge regression: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge  
bayesian ridge regression: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html#sklearn.linear_model.BayesianRidge  
lasso regression: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso  

Where fbProphet has been used: https://github.com/Not-A-Builder/SMP-Methodology/tree/master/fbprophet%20source%20code  
and the original code is present at: https://github.com/Not-A-Builder/prophet/blob/master/python/fbprophet/forecaster.py  

For short-term SMP, the following models have been used:
linear regression: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression  
svm: kernels (rbf, linar, polynommial) http://scikit-learn.org/stable/modules/svm.html#kernel-functions  
and svr: http://scikit-learn.org/stable/modules/svm.html#svr  
rnn: https://keras.io/layers/recurrent/#rnn  
mathematical models and diagrams: http://adventuresinmachinelearning.com/keras-lstm-tutorial/  

------------------------------------------------------------------------------------------------------------------------------

### 3: Diagrams of each model.

Plot diagrams and plots for each of the used models.

------------------------------------------------------------------------------------------------------------------------------

Regression: https://www.google.co.in/search?num=30&newwindow=1&source=lnms&tbm=isch&sa=X&ved=0ahUKEwjPoYH4mcTeAhWJRY8KHWjeAUkQ_AUIDygC&biw=1680&bih=948&q=regression&tbs=ift:png#imgrc=6GNeLioGbP3xtM:  
Ridge Regression: https://www.google.co.in/search?newwindow=1&biw=1680&bih=948&tbs=ift%3Apng&tbm=isch&sa=1&ei=PdzjW7DMCIuQvQSW0ZjYDQ&q=ridge+regression&oq=ridge+regression&gs_l=img.3..0l10.42751.43439.0.43853.6.6.0.0.0.0.194.477.0j3.3.0....0...1c.1.64.img..3.3.475...0i7i30k1j0i10i24k1.0.a2hAAqIdalE#imgrc=Eminvs2kOBRBZM:  
Bayesian Ridge Rgeression: https://www.google.co.in/search?newwindow=1&biw=1680&bih=948&tbs=ift%3Apng&tbm=isch&sa=1&ei=atzjW4nRA4XLvgSxqbK4Aw&q=bayesian+ridge+regression&oq=bayesian+ridge+regression&gs_l=img.3..0i24k1l3.52184.53530.0.54520.9.9.0.0.0.0.157.957.0j7.7.0....0...1c.1.64.img..2.2.272...0i30k1.0.YF7ncWFfOyY#imgrc=tOOn78QGLbz7tM:  
Lasso Regression: https://www.google.co.in/search?newwindow=1&biw=1680&bih=948&tbs=ift%3Apng&tbm=isch&sa=1&ei=atzjW4nRA4XLvgSxqbK4Aw&q=bayesian+ridge+regression&oq=bayesian+ridge+regression&gs_l=img.3..0i24k1l3.52184.53530.0.54520.9.9.0.0.0.0.157.957.0j7.7.0....0...1c.1.64.img..2.2.272...0i30k1.0.YF7ncWFfOyY#imgrc=sVZE7U2uZqTfsM:  

RNN: https://www.google.co.in/search?newwindow=1&biw=1680&bih=948&tbs=ift%3Apng&tbm=isch&sa=1&ei=Dd3jW5bcOsyMvQScxLRg&q=RNN+working&oq=RNN+working&gs_l=img.3..0i24k1.50664.54969.0.55181.19.14.1.3.3.0.218.1678.0j10j1.11.0....0...1c.1.64.img..4.14.1550...0j35i39k1j0i8i30k1j0i67k1j0i30k1.0.HecC81xFAbk#imgrc=T8xvOwKI6erKlM:  
SVM: https://en.wikipedia.org/wiki/Support_vector_machine#/media/File:SVM_margin.png  

------------------------------------------------------------------------------------------------------------------------------

### 4: Purpose of the models.

Why we are using Regression models for this problem.

Answer from https://en.wikipedia.org/wiki/Regression_analysis :  

------------------------------------------------------------------------------------------------------------------------------

In statistical terms, regression analysis modelling takes into account the entire collection of statistical methodologies that are employed to correctly estimate the correlation among multiple variables. The primary focus is on the the relationship between the depnedent variable in a particular problem, and, usually more than one, independent variables, which are also called the 'predictors', that is, the variables that will essentially help to predict the value or the effectiveness of the dependent variables. This analysis method called 'regression' will eventually help to understand the relationship between the 'criterion' variable, which is the dependent variable and the predictors. This will consequently help to understand how changing any one, or more, independent variable value, keeping the other values fixed, will affect the final predicted value of the dependent variable.  

So what a regression analysis does, is make an estimate with respect to the conditional expectations of the dependent variable, given the value(s) of the independent variables or the predictors. In mathematical terms, the regression analysis 
actually calculates the mean, or the average value of the dependent variable keeping the independent variable(s) fixed. The primary focus is on a given quantile, or any partciular location parameter, for that matter, of the conditional distribution of the dependent variable, with repsect to the one, or more, independent variables. In most cases however, the function is what is needed to be estimated, called the regression function - the function of the independent variables. 

A common problem that is solved by the regression analysis is the House Price problem. Suppose, for a particular city, we want to find the price of buying a house, given its area, location, age, number of floors and so on and so forth. So, here is a simple regression analysis for predicting the dependent variable - the price of the house, given the independent variables - the predictors - the area in square feet, the location in the city, the age of the house, the number of floors in the house et cetera, corresponding to the data in the collected dataset of historical data. So what regression analysis will do, is map the values of the independent variables, plotted in the x-axis of a 2-D curve, to particular values of the value to be predicted, to be plotted in the y-axis of the same curve. Then the algorithm for the regression analysis will find the best fit line that fits all the plotted data points in the plot. As a result you will have the best-fit line that fits to the historical data, and if you now want to make a prediction (for instance, what will be the price of the house, given its location, age, number of floors, and the floor area) then all you have to do is just draw a vertical line from the particular value(s) in the x-axis up to the plotted best-fit line, and then drop it onto the corresponding y-axis in order to get the associated price of that particluar house.  

For stock-market predictions, it is important to note, that a very similar case arises that can be  dealt with using regression analysis. For instance, if we deal with SP500 data for a given amount of time, such that the dataset comprises of the dates and the corresponding price associated with the security for that particular date. This can be plotted in a 2-D curve to get the datapoints from all the historical data present in the dataset, and after using regression analysis, we will be getting a best-fit line which will help us to make future prediction(s), with respect to the price of the security, based on the pattern that is generated from the existing historical data, given a particular date. Hence we have made use of regression analysis techniques in order to properly understand predictions and how the stock market behaves. Regression analysis best fits this form of learning.

------------------------------------------------------------------------------------------------------------------------------

### 5: Purpose of long-term/short-term SMP.

Stock market predictions are essentially what is the ruling factor of all the dynamics of every society. Stock market prices actually influence the social. political and, needless to say, economic dynamics of the world we live in. Hence predicting stock market values will essentially predict the future. But then, we cannot make predictions with accuracy that is actually useful. Suppose for instance, we get predictions of accuracy 0.8 and 0.9 respectively. Then what we actually take into account are the mis-predictions of values 0.2 and 0.1 respectively. Hence, since we cannot increase the accuracy, what we can actually do with various regression analysis techniques, is minimise the risk associated with these predictions. So the ultimate aim for stock market predictions is to minimise the risk and consequently inccrease the accuracy of the predictions.

### 6: Start LateX documentation.

# 10-05-18

### 1. Details about the prediction function needs to be understood

a. Analysis of the prediction function (.predict) for time-series data
    
b. The use of np.exp() is for the purpose of depicting a growth/decay in the curve. For more details please follow this link: https://mathinsight.org/exponential_function

##### Solution: https://github.com/Not-A-Builder/SMP-Methodology/tree/master/fbprophet%20source%20code

### 2. Do analysis with minimum 4 separate methodologies for time series data analysis

You can find adequate relevant data on the types of Time Series Analysis and Forecasting at the link: http://www.statgraphics.com/time-series-analysis-and-forecasting

a. Analyse and document their nature
    
b. Results need to be plotted in form of comparative analysis

##### Solution: https://github.com/Not-A-Builder/SMP-Methodology/blob/master/Long_term-Regression.ipynb

### 3. Document writings are to be planned in LateX platform.

Work to be done on: https://github.com/Not-A-Builder/LateX-Guides-and-Templates/blob/master/A%20LateX%20Guide/Template%20Paper/article.tex

##### Solution: https://github.com/Not-A-Builder/LateX-Guides-and-Templates



