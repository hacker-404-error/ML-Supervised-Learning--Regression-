# Supervised Machine Learning - Regression

- Supervised learning is the types of machine learning in which machines are trained using well "`labelled`" training data, and on basis of that data, machines predict the output. The labelled data means some input data is already tagged with the correct output.

- In supervised learning, the training data provided to the machines work as the supervisor that teaches the machines to predict the output correctly. It applies the same concept as a student learns in the supervision of the teacher.

- Supervised learning is a process of providing input data as well as correct output data to the machine learning model. The aim of a supervised learning algorithm is to find a mapping function to map the input variable(x) with the output variable(y).

## How Supervised Learning Works?

In supervised learning, models are trained using labelled dataset, where the model learns about each type of data. Once the training process is completed, the model is tested on the basis of test data (a subset of the training set), and then it predicts the output.

The working of Supervised learning can be easily understood by the below example and diagram:

<br>
<div align="center">
    <img src="https://i.ibb.co/MNWk6xv/supervised-machine-learning.jpg" alt="how supervised learning work?">
</div>
<br>

## Steps Involved in Supervised Learning:
- First Determine the type of training dataset.
- Collect/Gather the labelled training data.
- Split the training dataset into training dataset, test dataset, and validation dataset.
- Determine the input features of the training dataset, which should have enough knowledge so that the model can accurately predict the output.
- Determine the suitable algorithm for the model, such as support vector machine, decision tree, etc.
- Execute the algorithm on the training dataset. Sometimes we need validation sets as the control parameters, which are the subset of training datasets.
- Evaluate the accuracy of the model by providing the test set. If the model predicts the correct output, which means our model is accurate.

# Types of supervised Machine learning Algorithms:

<br>
<div align="center">
    <img src="https://media.geeksforgeeks.org/wp-content/uploads/SL-type.png" alt="Types of supervised learning">
</div>
<br>



# Regression

- Regression analysis is a statistical method to model the relationship between a `dependent (target)` and `independent (predictor)` variables with one or more independent variables.
- In Regression, we plot a graph between the variables which best fits the given datapoints, using this plot, the machine learning model can make predictions about the data. In simple words, "`Regression shows a line or curve that passes through all the datapoints on target-predictor graph in such a way that the vertical distance between the datapoints and the regression line is minimum.`" The distance between datapoints and line tells whether a model has captured a strong relationship or not.

# Terminologies Related to the Regression Analysis:
- `Dependent Variable`: The main factor in Regression analysis which we want to predict or understand is called the dependent variable. It is also called target variable.
- `Independent Variable`: The factors which affect the dependent variables or which are used to predict the values of the dependent variables are called independent variable, also called as a predictor.
- `Outliers`: Outlier is an observation which contains either very low value or very high value in comparison to other observed values. An outlier may hamper the result, so it should be avoided.
- `Multicollinearity`: If the independent variables are highly correlated with each other than other variables, then such condition is called Multicollinearity. It should not be present in the dataset, because it creates problem while ranking the most affecting variable.
- `Underfitting and Overfitting`: If our algorithm works well with the training dataset but not well with test dataset, then such problem is called Overfitting. And if our algorithm does not perform well even with training dataset, then such problem is called underfitting.

## Types of Regression

 - #### Simple Linear Regression
 - #### Multiple Linear Regression
 - #### Polynomial Regression
 - #### Support Vector Regression
 - #### Decision Tree Regression
 - #### Random Forest Regression

# Simple Linear Regression

The Simple Linear Regression model can be represented using the below equation:
```
y= a0 + a1x + ε 
```
Where,
- a0= It is the intercept of the Regression line (can be obtained putting x=0)
- a1= It is the slope of the regression line, which tells whether the line is increasing or decreasing.
- ε = The error term. (For a good model it will be negligible)
  
## Implementation of Simple Linear Regression Algorithm using Python

`Problem Statement example for Simple Linear Regression:`

Here we are taking a dataset that has two variables: salary (dependent variable) and experience (Independent variable). 

|     | YearsExperience | Salary   |
| --- | --------------- | -------- |
| 0   | 1.1             | 39343.0  |
| 1   | 1.3             | 46205.0  |
| 2   | 1.5             | 37731.0  |
| 3   | 2.0             | 43525.0  |
| 4   | 2.2             | 39891.0  |
| 5   | 2.9             | 56642.0  |
| ... | ...             | ...      |
| 25  | 9.0             | 105582.0 |
| 26  | 9.5             | 116969.0 |
| 27  | 9.6             | 112635.0 |
| 28  | 10.3            | 122391.0 |
| 29  | 10.5            | 121872.0 |

The goals of this problem is:
- We want to find out if there is any correlation between these two variables
- We will find the best fit line for the dataset.
- How the dependent variable is changing by changing the independent variable.

## Step-1: Data Pre-processing
[Data Pre-processing](https://github.com/hacker-404-error/ML-Data-Preprocessing) 
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hacker-404-error/ML-Data-Preprocessing/blob/master/data_preprocessing_tools.ipynb)

## Step-2: Fitting the Simple Linear Regression to the Training Set:

we will import the LinearRegression class of the linear_model library from the scikit learn. After importing the class, we are going to create an object of the class named as a regressor.
```
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```
## Step: 3. Prediction of test set result:

We will create a prediction vector y_pred, and x_pred, which will contain predictions of test dataset, and prediction of training set respectively.
```
#Prediction of Test and Training set result  
y_pred= regressor.predict(x_test)  
x_pred= regressor.predict(x_train)  
```
## Step: 4. visualizing the Training set results:

- we will use the scatter() function of the pyplot library, which we have already imported in the pre-processing step. The scatter () function will create a scatter plot of observations.

- In the x-axis, we will plot the Years of Experience of employees and on the y-axis, salary of employees. 

- Now, we need to plot the regression line, so for this, we will use the plot() function of the pyplot library. In this function, we will pass the years of experience for training set, predicted salary for training set x_pred, and color of the line(Blue).

- Next, we will give the title for the plot. So here, we will use the title() function of the pyplot library and pass the name ("Salary vs Experience (Training Dataset)".

- After that, we will assign labels for x-axis and y-axis using xlabel() and ylabel() function.
```
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, x_pred, color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()   
``` 
## Step: 5. visualizing the Test set results:
```
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_pred, color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
```