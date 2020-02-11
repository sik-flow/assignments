
# Module 5 Assessment 

Welcome to your Module 5 Assessment. You will be tested for your understanding of concepts and ability to programmatically solve problems that have been covered in class and in the curriculum. 

**_Read the instructions very carefully!_** You will be asked both to write code and respond to a few short answer questions.  

The goal here is to demonstrate your knowledge. Showing that you know things about certain concepts and how to apply different methods is more important than getting the best model.

The sections of the assessment are:

- Decision Trees
- Ensemble Models 
- PCA
- Clustering

**Note on the short answer questions**: 
> Please use your own words, even if you consult another source to help you craft your response. Short answer questions are not necessarily being assessed on grammatical correctness or sentence structure, but do your best to communicate your answers clearly!


## Decision Trees [Suggested Time: 15 min]

### Concepts 
You're given a dataset of **30** elements, 15 of which belong to a positive class (denoted by *`+`* ) and 15 of which do not (denoted by `-`). These elements are described by two attributes, A and B, that can each have either one of two values, true or false. 

The diagrams below show the result of splitting the dataset by attribute: the diagram on the left hand side shows that if we split by Attribute A there are 13 items of the positive class and 2 of the negative class in one branch and 2 of the positive and 13 of the negative in the other branch. The right hand side shows that if we split the data by Attribute B there are 8 items of the positive class and 7 of the negative class in one branch and 7 of the positive and 8 of the negative in the other branch.

<img src="images/decision_stump.png">

**1.1) Which one of the two attributes resulted in the best split of the original data? How do you select the best attribute to split a tree at each node?** _(Hint: Mention splitting criteria)_


```
# Attribute A generates the best split for the data. 
# The best attribute to split a tree at each node is selected by considering 
# the attribute that creates the purest child nodes. Gini impurity and information 
# gain are two criteria that can be used to measure the quality of a split.
```

### Decision Trees for Regression 

In this section, you will use decision trees to fit a regression model to the Combined Cycle Power Plant dataset. 

This dataset is from the UCI ML Dataset Repository, and has been included in the `data` folder of this repository as an Excel `.xlsx` file, `Folds5x2_pp.xlsx`. 

The features of this dataset consist of hourly average ambient variables taken from various sensors located around a power plant that record the ambient variables every second.  
- Temperature (AT) 
- Ambient Pressure (AP) 
- Relative Humidity (RH)
- Exhaust Vacuum (V) 

The target to predict is the net hourly electrical energy output (PE). 

The features and target variables are not normalized.

In the cells below, we import `pandas` and `numpy` for you, and we load the data into a pandas DataFrame. We also include code to inspect the first five rows and get the shape of the DataFrame.


```
import pandas as pd 
import numpy as np 

# Load the data
filename = 'data/Folds5x2_pp.xlsx'
df = pd.read_excel(filename)
```


```
# Inspect the first five rows of the dataframe
df.head()
```


```
# Get the shape of the dataframe 
df.shape
```

Before fitting any models, you need to create training and testing splits for the data.

Below, we split the data into features and target ('PE') for you. 


```
X = df[df.columns.difference(['PE'])]
y = df['PE']
```

**1.2) Split the data into training and test sets. Create training and test sets with `test_size=0.5` and `random_state=1`.** 


```
# Include relevant imports 
from sklearn.model_selection import train_test_split

# Create training and test sets with test_size=0.5 and random_state=1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
```

**1.3) Fit a vanilla decision tree regression model with scikit-learn to the training data.** Set `random_state = 1` for reproducibility. **Evaluate the model on the test data.** 


```
# Bring in necessary imports 
from sklearn.tree import DecisionTreeRegressor

# Fit the model to the training data 
dt = DecisionTreeRegressor(random_state=1)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)
```

**1.4) Obtain the mean squared error, mean absolute error, and coefficient of determination (r2 score) of the predictions on the test set.** _Hint: Look at the `sklearn.metrics` module._


```
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))
```

Hint: MSE = 22.21041691053512

### Hyperparameter Tuning of Decision Trees for Regression

For this next section feel free to refer to the scikit learn documentation on [decision tree regressors](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)

**1.5) Add hyperparameters to a a new decision tree and fit it to our training data and evaluate the model with the test data.**


```
# Evaluate the model on test data 
dt_tuned = DecisionTreeRegressor(
    random_state=1,
    max_depth=3,
    min_samples_leaf=2,
)
dt_tuned.fit(X_train,y_train)
y_pred_tuned = dt_tuned.predict(X_test)
```

**1.6) Obtain the mean squared error, mean absolute error, and coefficient of determination (r2 score) of the predictions on the test set. Did this improve your previous model? (It's ok if it didn't)**


```

# Example: adjusting the max depth changes how many splits can happen on a single branch.
# Setting this to three helped improve the model and reduced overfitting.

print("Mean Squared Error:", mean_squared_error(y_test, y_pred_tuned))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_tuned))
print("R-squared:", r2_score(y_test, y_pred_tuned))
```

## Ensemble Methods [Suggested Time: 20 min]

### Introduction to Ensemble Methods

**2.1) Explain how the random forest algorithm works. Why are random forests resilient to overfitting?**

_Hint: Your answer should discuss bagging and the subspace sampling method._


```

# A random forest is made up of many decision trees that are each trained 
# on different samples of the data, where the data used to train each decision 
# tree is sampled with replacement from the training data. Then, a randomly 
# selected subset of features are used as predictors for each node for each one 
# of the decision trees, instead of using all available features. This is called 
# the subspace sampling method.  

# The resulting model has a collection of decision trees that have been trained 
# on different samples of data looking at different subsets of parameter space, 
# which makes it resilient to the effects of noisy data, and thus resilient to 
# over-fitting.
```

### Random Forests and Hyperparameter Tuning using GridSearchCV

In this section, you will perform hyperparameter tuning for a Random Forest classifier using GridSearchCV. You will use `scikit-learn`'s wine dataset to classify wines into one of three different classes. 

After finding the best estimator, you will interpret the best model's feature importances. 

In the cells below, we have loaded the relevant imports and the wine data for you. 


```
# Relevant imports 
from sklearn.datasets import load_wine

# Load the data 
wine = load_wine()
X, y = load_wine(return_X_y=True)
X = pd.DataFrame(X, columns=wine.feature_names)
y = pd.Series(y)
y.name = 'target'
df = pd.concat([X, y.to_frame()], axis=1)
```

In the cells below, we inspect the first five rows of the dataframe and compute the dataframe's shape.


```
df.head()
```


```
df.shape
```

We also get descriptive statistics for the dataset features, and obtain the distribution of classes in the dataset. 


```
X.describe()
```


```
y.value_counts().sort_index()
```

You will now perform hyper-parameter tuning for a Random Forest classifier.

**2.2) Construct a `param_grid` dictionary to pass to `GridSearchCV` when instantiating the object. Choose at least 3 hyper-parameters to tune and 3 values for each.** 


```
#this is only an example (student's answers will likely be different)
param_grid = { 
    'n_estimators': [5,10,15,20],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6],
    'criterion' :['gini', 'entropy']}
```

Now that you have created the `param_grid` dictionary of hyperparameters, let's continue performing hyperparameter optimization of a Random Forest Classifier. 

In the cell below, we include the relevant imports for you.


```
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
```

**2.3) Create an instance of a Random Forest classifier estimator; call it `rfc`.** Make sure to set `random_state=42` for reproducibility. 


```
rfc = RandomForestClassifier(random_state=42)
```

**2.4) Create an instance of an `GridSearchCV` object and fit it to the data.** Call the instance `cv_rfc`. 

* Use the random forest classification estimator you instantiated in the cell above, the parameter grid dictionary constructed, and make sure to perform 5-fold cross validation. 
* The fitting process should take 10 - 15 seconds to complete. 


```
# Create an instance of a `GridSearchCV` object with the appropriate params. 
cv_rfc = GridSearchCV(estimator=rfc, 
                      param_grid=param_grid, 
                      cv = 5)

# Fit it to the data
cv_rfc.fit(X, y)
```

**2.5) What are the best training parameters found by GridSearchCV?** 

_Hint: Explore the documentation for GridSearchCV._ 


```
cv_rfc.best_params_
```

In the cell below, we create a variable `best_model` that holds the best model found by the grid search.


```
best_model = cv_rfc.best_estimator_
```

Next, we give you a function that creates a horizontal bar plot to visualize the feature importances of a model, sorted in descending order. 


```
import matplotlib.pyplot as plt 
%matplotlib inline 

def create_plot_of_feature_importances(model, X):
    ''' 
    Inputs: 
    
    model: A trained ensemble model instance
    X: a dataframe of the features used to train the model
    '''
    
    feat_importances = model.feature_importances_

    features_and_importances = zip(X.columns, feat_importances)
    features_and_importances = sorted(features_and_importances, 
                                     key = lambda x: x[1], reverse=True)
    
    features = [i[0] for i in features_and_importances]
    importances = [i[1] for i in features_and_importances]
    
    plt.figure(figsize=(10, 6))
    plt.barh(features, importances)
    plt.gca().invert_yaxis()
    plt.title('Feature Importances')
    plt.xlabel('importance')
```

**2.6) Create a plot of the best model's feature importances.** 

_Hint: To create the plot, pass the appropriate parameters to the function above._


```
create_plot_of_feature_importances(best_model, X)
```

**2.7) What are this model's top 3 features in order of descending importance?**


```
# flavanoids, color_intensity, alcohol
# Note: this may vary depending how the student tuned the model
```

## Principal Components Analysis [Suggested Time: 20 min]

### Training a model with PCA-extracted features

In this section, you'll apply the unsupervised learning technique of Principal Components Analysis to the wine dataset. 

You'll use the principal components of the dataset as features in a machine learning model. You'll use the extracted features to train a vanilla Random Forest Classifier, and compare model performance to a model trained without PCA-extracted features. 

In the cell below, we import the data for you, and we split the data into training and test sets. 


```
from sklearn.datasets import load_wine
X, y = load_wine(return_X_y=True)

wine = load_wine()
X = pd.DataFrame(X, columns=wine.feature_names)
y = pd.Series(y)
y.name = 'class'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

**3.1) Fit PCA to the training data.** 

Call the PCA instance you'll create `wine_pca`. Set `n_components=0.9` and make sure to use `random_state = 42`.

_Hint: Make sure to include necessary imports for **preprocessing the data!**_


```
# Relevant imports 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Scale the data 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)

# Create and fit an instance of PCA. Call it `wine_pca`. 
wine_pca = PCA(n_components = 0.9, random_state=42)
wine_pca.fit(X_train_scaled)
```

**3.2) How many principal components are there in the fitted PCA object?**

_Hint: Look at the list of attributes of trained `PCA` objects in the scikit-learn documentation_


```
print(wine_pca.n_components_)
```

*Hint: you should end up with 8 components.*

Next, you'll reduce the dimensionality of the training data to the number of components that explain at least 90% of the variance in the data, and then you'll use this transformed data to fit a Random Forest classification model. 

You'll compare the performance of the model trained on the PCA-extracted features to the performance of a model trained using all features without feature extraction.

**3.3) Transform the training features into an array of reduced dimensionality using the `wine_pca` PCA object you've fit in the previous cell.** Call this array `X_train_pca`.


```
X_train_pca = wine_pca.transform(X_train_scaled)
```

Next, we create a dataframe from this array of transformed features and we inspect the first five rows of the dataframe for you. 


```
# Create a dataframe from this array of transformed features 
X_train_pca = pd.DataFrame(X_train_pca)

# Inspect the first five rows of the transformed features dataset 
X_train_pca.head()
```

#### You will now use the PCA-extracted features to train a random forest classification model.

**3.4) Instantiate a vanilla Random Forest Classifier (call it `rfc`) and fit it to the transformed training data.** Set `random_state = 42`. 


```
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train_pca, y_train)
```

**3.5) Evaluate model performance on the test data and place model predictions in a variable called `y_pca_pred`.**

_Hint: Make sure to transform the test data the same way as you transformed the training data!!!_


```
# Scale the test data using the `scaler` object 
X_test_scaled = scaler.transform(X_test)

# Transform the scaled test data using the `wine_pca` object
X_test_pca = wine_pca.transform(X_test_scaled)
X_test_pca = pd.DataFrame(X_test_pca)

# Evaluate model performance on transformed test data
y_pca_pred = rfc.predict(X_test_pca)
```

In the cell below, we print the classification report for the model performance on the test data. 


```
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pca_pred))
```

Run the cell below to fit a vanilla Random Forest Classifier to the untransformed training data,  evaluate its performance on the untransformed test data, and print the classification report for the model. 


```
vanilla_rfc = RandomForestClassifier(random_state=42)
vanilla_rfc.fit(X_train, y_train)

y_pred = vanilla_rfc.predict(X_test)

print(classification_report(y_test, y_pred))
```

**3.6) Compare model performance. Did the overall accuracy of the model improve when using the transformed features?**


```
# The model accuracy for the model trained using the PCA-extracted features increased
# relative to the model trained using the untransformed features. 
```

## Clustering [Suggested Time: 30 min]

### Clustering Algorithms: k-means and hierarchical agglomerative clustering

#### 4.1) Using the gif below for reference, describe the steps of the k-means clustering algorithm.
* If the gif doesn't run, you may access it via [this link](images/centroid.gif).

<img src='images/centroid.gif'>


```

# Steps of the k-means clustering algorithm: 
# 1. To start, k starting "mean" points are randomly generated. 
# 2. Then, each observation point is assigned to the "mean" point closest to it. 
# 3. The centroid of each one of the k clusters becomes the new "mean". 
# Steps 2 and 3 are repeated until the centroids move no more than an accepted
# tolerance. 


```

#### 4.2) In a similar way, describe the process behind Hierarchical Agglomerative Clustering.


```

# In hierarchical agglomerative clustering, all individual points start as their own clusters.
# Clusters are grown by merging individual points following some criteria (for example, 
# points closest to each other are merged into a single cluster), until some end point is reached. 
```

### k-means Clustering

For this question, you will apply k-means clustering to your now friend, the wine dataset. 

You will use scikit-learn to fit k-means clustering models, and you will determine the optimal number of clusters to use by looking at silhouette scores. 

We load the wine dataset for you in the cell below. 


```
from sklearn.datasets import load_wine

X, y = load_wine(return_X_y=True)
wine = load_wine()
X = pd.DataFrame(X, columns = wine.feature_names)
```

**4.3) Write a function called `get_labels` that will find `k` clusters in a dataset of features `X`, and return the labels for each row of `X`.**

_Hint: Within the function, you'll need to:_
* instantiate a k-means clustering model (use `random_state = 1` for reproducibility),
* fit the model to the data, and
* return the labels for each point.


```
# Relevant imports 
from sklearn.cluster import KMeans

def get_labels(k, X):
    
    # Instantiate a k-means clustering model with random_state=1 and n_clusters=n 
    kmeans = KMeans(n_clusters=k, random_state=1)
    
    # Fit the model to the data 
    kmeans.fit(X)
    
    # Return the predicted labels for each row in the data
    return kmeans.labels_
```

**4.4) Fit the k-means algorithm to the wine data for k values in the range 2 to 9 using the function you've written above. Obtain the silhouette scores for each trained k-means clustering model, and place the values in a list called `silhouette_scores`.** 

We have provided you with some starter code in the cell below.

_Hints: What imports do you need? Do you need to pre-process the data in any way before fitting the k-means clustering algorithm?_ 


```
# Relevant imports 
from sklearn.metrics import silhouette_score

# Preprocessing is needed. Scale the data.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create empty list for silhouette scores 
silhouette_scores= []

for k in range(2, 10):
    labels = get_labels(k, X_scaled)
    
    score = silhouette_score(X_scaled, labels, metric='euclidean')
    
    silhouette_scores.append(score)
```

Run the cell below to plot the silhouette scores obtained for each different value of k against k, the number of clusters we asked the algorithm to find. 


```
plt.plot(range(2, 10), silhouette_scores, marker='o')
plt.title('Silhouette scores vs number of clusters')
plt.xlabel('k (number of clusters)')
plt.ylabel('silhouette score')
```

**4.5) Which value of k would you choose based on the plot of silhouette scores? How does this number compare to the number of classes in the wine dataset?**

Hint: this number should be <= 5.  If it's not, check your answer in the previous section.


```

# We obtain the best value of the silhouette score for k = 3. 
# This happens to be equal to the number of classes in the wine dataset! 
```
