---
layout: post
title: "Using Pipelines and Gridsearch in Scikit-Learn"
date: 2018-09-11
tag: [pipelines, modeling, sklearn, scikit-learn]
image:"/assets/img/pipelines/pipeline.png"
excerpt: "An introduction to pipelines and gridsearching in the scikit-learn library. Explanation of pipelines and gridsearch and codealong included"
---
![Jumbled mess of overlaid pipes. Also known as my attempt to be funny]({{'/assets/img/pipelines/pipeline.png'}})

When modeling with data, we often have to go through several steps to transform the data before we are able to model it. How exactly we will transform the data depends on what exacly we are attempting to model. If we are working with text data, we may want to use something like `CountVectorizer` in order to get counts of each word in each document. If we are working with categorical features, we may want to create one-hot encoded features to represent each possible category. If we are working with timeseries data, we may want to convert our data to be a datetime object.

All of these transformations are an important part of the EDA and data cleaning process. However, manually completing each transfomration can be confusing and frankly difficult. Luckily for us, `Pipeline` is a wonderful module in the scikit-learn library that makes this process of applying transformations much easier. Let's go through an example of how to use pipelines below.

---
Before starting, you may need to install the scikit-learn or pandas libraries. You can run the following code from your terminal to do so


```python
pip install sklearn
pip install pandas
```

If you are using a Jupyter Notebook, you can run this code from the notebook by simply prepending an exclamation point like this


```python
!pip install sklearn
!pip install pandas
```

---
Now let's jump into our example. For this, I will be using some posts that I scraped from reddit using both the [reddit api]('https://www.reddit.com/dev/api/') and [PRAW]('https://praw.readthedocs.io/en/latest/'). The csv file with the data used in this post is available [here]({{'/assets/data/reddit_posts.csv'}}). In order to follow along, download the .csv file from the link above and place the file into the same repo (folder) as your python file.

First, let's run all of the necessary imports:


```python
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
```

Now that we've imported everything we need, we will use pandas to load in our reddit data as a dataframe.


```python
df = pd.read_csv('./reddit_posts.csv')
df.head()
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
      <th>Unnamed: 0</th>
      <th>gilded</th>
      <th>num_comments</th>
      <th>num_crossposts</th>
      <th>ups</th>
      <th>subreddit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>64</td>
      <td>5117</td>
      <td>10</td>
      <td>139149</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>844</td>
      <td>1</td>
      <td>20255</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>4745</td>
      <td>0</td>
      <td>91308</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>1718</td>
      <td>0</td>
      <td>19882</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2</td>
      <td>1948</td>
      <td>5</td>
      <td>89392</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



**If the code above does not work for you, make sure that you've moved the reddit .csv file into the same folder where your python file exists.**

You may notice that we have duplicated the index of our dataframe with the `Unnamed: 0` column. We'll drop that column in the cell below.


```python
# axis=1 means look for the label among the columns
# inplace=True means that the change will occur at the original object rather than outputting a copy
df.drop('Unnamed: 0', axis=1, inplace=True)

df.head()
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
      <th>gilded</th>
      <th>num_comments</th>
      <th>num_crossposts</th>
      <th>ups</th>
      <th>subreddit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>64</td>
      <td>5117</td>
      <td>10</td>
      <td>139149</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>844</td>
      <td>1</td>
      <td>20255</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>4745</td>
      <td>0</td>
      <td>91308</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1718</td>
      <td>0</td>
      <td>19882</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>1948</td>
      <td>5</td>
      <td>89392</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



The features in this dataframe are:

| Feature          | Type      | Description                                                |
|------------------|-----------|------------------------------------------------------------|
| `subreddit`      | Target    | 1 or 0 to represent which subreddit this post is from      |
| `gilded`         | Predictor | Number of times post received reddit gold                  |
| `num_comments`   | Predictor | Number of comments on the thread                           |
| `num_crossposts` | Predictor | Number of times thread was crossposted in other subreddits |
| `ups`            | Predictor | Number of upvotes this post received                       |

---
For this example, we will be building a classification model using logistic regression. Before we create our model, let's first create our X and y variables and then train/test split our data


```python
X = df[['gilded', 'num_comments', 'num_crossposts', 'ups']]
y = df['subreddit']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```

Remember that when using logistic regression through the scikit-learn library, there is built in regularization. Since we are regularizing our data, we first have to scale it. Without using pipelines, the remainder of our code would probably look something like this


```python
# Create scaler object
ss = StandardScaler()

# Fit scaler to training data
ss.fit(X_train)

# Transform train and test X data and save as new variables
X_train_scaled = ss.transform(X_train)
X_test_scaled = ss.transform(X_test)

# Instantiate our logistic regression object
logreg = LogisticRegression()

# Fit logistic regression to scaled training data
logreg.fit(X_train_scaled, y_train)

# Score logistic regression on scaled testing data
logreg.score(X_test_scaled, y_test)
```




    0.8524590163934426



With this relatively simple example, the preceding code is not too bad. However, using pipelines can greatly simplify the process. Pipelines act as a blueprint for transforming your data and fitting a given model. When instantiating a pipeline, there are two parameters, steps and memory. 

The steps parameter is a list of what will happen to data that enters the pipeline. Each individual step is entered as a tuple where the first item is the name of the step, and the second item is the transfomer. An example step might be `('lr', LinearRegression())`, where `'lr'` is an arbitrary name for the linear regression model. The very last step must be an estimator, meaning that it must be a class that implements a `.fit()` method. All other steps must be transformers, meaning they must implement a `.fit()` method and a `.transform()` method. While you may use any transformer and estimator from the scikit-learn library, you may also write your own custom classes as long as they can implement the appropriate methods.

The model creation, fitting, and scoring process is accomplished below using pipelines


```python
# Create pipeline with steps as list of tuples
pipeline = Pipeline([
    ('ss', StandardScaler()), # tuple is (name, Transformer)
    ('logreg', LogisticRegression())
])

# Fit pipeline on training data
pipeline.fit(X_train, y_train)

# Score pipeline on testing data
pipeline.score(X_test, y_test)
```




    0.8524590163934426



Just like the earlier code, our pipeline will first use a StandardScaler object to scale whatever data enters the pipeline, and then will use a logistic regression model to either fit or score the data, depending on which function is called. With the pipeline, we no longer have to store our transformed predictor matrices as their own variables, the pipeline will handle this process for us. This also helps to eliminate one of my more common errors, accidentally fitting and scoring a model on the unscaled or non-transformed predictor matrix. 

---
Pipelines become more and more powerful when you realize that you can apply multiple transformers before fitting your model in the last step. Let's say you also wanted to look at all of the interaction terms between your features in the ealier example using `PolynomialFeatures`. Doing so is quite straightforward with a pipeline.


```python
# Create the pipeline object
polynomial_pipeline = Pipeline([
    ('poly', PolynomialFeatures()),
    ('ss', StandardScaler()),
    ('logreg', LogisticRegression())
])

# Fit the pipeline on the training data
polynomial_pipeline.fit(X_train, y_train)

# Score the pipeline on the testing data
polynomial_pipeline.score(X_test, y_test)
```




    0.8668032786885246



Applying more transformers is as simple as adding in new tuples to the list of steps in the pipeline. For those feeling ambitious, feel free to check out the `FeaturesUnion` module that can be used to combine the output of multiple individual pipelines

---
### GridSearch

After fitting our model above, we may be wondering if that is the best possible logistic regression model for the data. After all, we may get better results by using stronger regularization or changing the regularization penalty. But if we wanted to check we would have to try using a bunch of different models...or we could use scikit-learn's `GridSearchCV`.

`GridSearchCV` is a scikit-learn module that allows you to programatically search for the best possible hyperparameters for a model. By passing in a dictionary of possible hyperparameter values, you can search for the combination that will give the best fit for your model. Grid search uses cross validation to determine which set of hyperparameter values will likely perform best on unseen testing data. By default, it uses three fold validation, although this number can be overwritten when a grid search object is instantiated.

Grid search requires two parameters, the estimator being used and a `param_grid`. The `param_grid` is a dictionary where the keys are the hyperparameters being tuned and the values are tuples of possible values for that specific hyperparameter. Writing all of this together can get a little messy, so I like to define the param_grid as a variable outside of the `GridSearchCV` object and just pass in the created variable. Below is an example of instantiating `GridSearchCV` with a logistic regression estimator.


```python
# Create the parameter dictionary for the param_grid in the grid search
parameters = {
    'C': (0.1, 1, 10),
    'penalty': ('l1', 'l2')
}

# Instantiate the gridsearch object with a Logistic Regression estimator and the 
# parameter dictionary from ealier as a param_grid
gs = GridSearchCV(LogisticRegression(), parameters)

```




    GridSearchCV(cv=None, error_score='raise',
           estimator=LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False),
           fit_params=None, iid=True, n_jobs=1,
           param_grid={'C': (0.1, 1, 10), 'penalty': ('l1', 'l2')},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=0)



This grid search object can now be used just like any other scikit-learn model. We can call `.fit()` and `.score()` as we see in the cell below


```python
# Fit the grid search model to the training data
gs.fit(X_train_scaled, y_train)

# Score the grid search model with the testing data
gs.score(X_test_scaled, y_test)
```




    0.8627049180327869



We can also use some of the attributes to see what the best parameters are.


```python
# Use best_estimator_ to return estimator with highest cv score
gs.best_estimator_
```




    LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l1', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



Using `.best_estimator_` will give us the estimator with the best cross validated score, but it is hard to read. We can use `.best_params_` for a more readable version of the best parameters for our model.


```python
gs.best_params_
```




    {'C': 0.1, 'penalty': 'l1'}



We can also look at each of the individual cross validated scores using the `.cv_results_` attribute, although you wouldn't be likely to do so unless you are digging into the details to figure out why you got an unexpected result. A more frequently used attribute is `.best_score_`, which will return the mean score from the three fold cross validation of the model with the best parameters.


```python
gs.best_score_
```




    0.8939808481532148



An important note of caution, it may be tempting to give your gridsearch a huge set of parameters to search over. Don't go overboard!!! Remember that each additional hyperparameter value adds more models that have to be fit. If you were tweaking 3 hyperparameters and passed in 20 possible values for each parameter, your grid search would have to fit:

$20 * 20 * 20 * 3 = 24,000$ models

While this may go quickly for some smaller models, as you work with more and more complex data this can start taking extremely long amounts of time. Instead, try searching with just a few parameters and then adjust values as necessary

---
### Combining Pipelines and Gridsearch

These two tools together can be incredibly powerful. Let's go back to our earlier example with the `polynomial_pipeline`. Remember that in this pipeline, we first used `PolynomialFeatures` to create interaction terms, and then used `StandardScaler` to scale our data. Using grid search, we can change hyperparameters of our transformers as well as our estimator. 

Because there are multiple different objects that we are changing the hyperparameters for, we do have to set up our param_grid slighly differently. Now instead of just using the hyperparameters we are tuning as keys, we have to follow the format `object name__hyperparameter name`. Note there is a double underscore between the object name and the hyperparameter name

The cell below shows an example of searching over different possibilities to get the best possible logistic regression model.


```python
# Create new parameter dictionary
grid_params = {
    # Key = step name from pipeline + __ + hyperparameter, value = tuple of possible values
    'poly__interaction_only': (True, False),
    'poly__include_bias': (True, False),
    'logreg__penalty': ('l1', 'l2'),
    'logreg__C': (0.01, 0.1, 1, 10),
    'logreg__fit_intercept': (True, False)
}

# Instantiate new gridsearch object
gs_2 = GridSearchCV(polynomial_pipeline, grid_params)

# Fit model to our training data
gs_2.fit(X_train, y_train)

# Score the model on our testing data
gs_2.score(X_test, y_test)
```




    0.9016393442622951




```python
gs_2.best_params_
```




    {'logreg__C': 10,
     'logreg__fit_intercept': True,
     'logreg__penalty': 'l1',
     'poly__include_bias': True,
     'poly__interaction_only': False}



There you have it. By tuning multiple hyperparameters across our transformers and estimator, we were able to get our best accuracy score yet of just over 90%.

Pipelines and GridSearch make an awesome combo, just remember not to overload your grid search with too many potential hyperparameter values. Happy modeling!
