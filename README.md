# ML
A simple implementation of Linear Regression from scratch. Basically a fancy hello world in ML, and to understand how it (LR, cost func, GD) works.


# Linear Regression implementation
This is (a supervised learning also) basically used to predict continuous value(s) like sale figures, etc.

The basic LR is of the form **`y = wx + b`**
*Here, w is the weight (coefficient of x) and b is the bias.*

***X***- The independent variable.

***Y***- The dependent variable (the model will predict Y values). 

***Weight (w)-*** The coefficient for the independent variable X. In machine learning lingo, we call coefficient(s) of X as _weights_.

***Bias (b)*** -The Y-intercept. In ML, we'll call this Bias. This essentially offsets all the predicted values of Y.

<hr>

### The data:
I have taken some random values to represent the **X** and **Y**. (The model will predict Y values). We will try and find the best fit line for this data.

| X | Y |
|---|---|
| 1.0  | 2.5  |
|  2.0 | 3.1  |
| 4.3 | 3.9  |
| 3.1 |  4.2 |
| 5.2  | 5.3  |
| 6.6 |  7.1 |
 
![alt text](/raw_data.png)

The algorithm will try and find (and optimise) the weight and bias. At the end, we should have an equation that represents the best fit line for this data set. In other words, it would predict the value of Y, given a value of X.

## Cost Function
Mean Squared Error (MSE) measures the average squared difference the actual and the predicted values. The output is a single number representing the cost, or score (associated with the current weight(s) and bias). The goal is to minimize MSE to improve the accuracy of the model.

Low error = Accurate results.

_(Our SLE equation is: y = mx + b)_



$$
MSE = \frac{1}{N}\sum_{i=1}^n (y_i - (mx_i + b))^2
$$

Here, 
>m, and b =  weight, and bias respectively. 


>N = number of observations/data points (on an axis).


>The $\frac{1}{N}\sum_{i=1}^n$ part is the mean.

## Gradient Descent

To minimize the MSE, we'll use Gradient Descent to calculate the gradient of our cost function. Using the derivative of the cost function (w.r.t weight and bias both) to find the gradient and then changing the weight, (and/or) bias to move in the direction opposite of the gradient (since it is a gradient, and we want to decrease our error, so we "descent the gradient").

The derivate(s) (gradient) of our MSE function are:

$w.r.t. 'm' (weight) = \frac{1}{N}\sum-2x_i(y_i - (mx_i + b))$

$w.r.t. 'b' (bias) = \frac{1}{N}\sum-2(y_i - (mx_i + b))$

## The model and results
The algorithm/model improves the pridiction with each iteration (optimises wt and bias) moving towards the direction suggested by the slope of the cost function (the gradient).

![initial state](/zero%20iter.png)
![10th iter](/10th%20iter.png)
![final res](/final%20result.png)

## Comparing the results with that of the LR model in the sklearn library.
The $R^2$ of sklearn LR model is: 0.6374...

The $R^2$ of sklearn LR model is: 0.6374...

>~63.74% of variance in the dependent variable Y is explained by the change in X. 
>Or, in simpler terms, how well the data fits the regression line. 

Below are the best fit line and predicted Y values from both the models-
![final sklearn](/final%20result_sklearn.png)
![initial my model](/final%20result_mym.png)
