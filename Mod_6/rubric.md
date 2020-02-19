## Natural Language Processing

#### 1)



Point Value : 2

0.75 point(s) : initialize TfidfVectorizer class

0.25 : remove stopwords

0.25 point(s) : train-test split before fitting vectorizer

0.5 point(s) : Fit-transform tf-idf vectorizer with train data

0.25 point(s) : Fit tf-idf vectorizer with test data

_Learning Goal : Use sci-kit learn text vectorizers to fit and transform text data into a format to be used in a ML model_

***
#### 2)


Point Value: 1

0.5 point(s) : fit classifier with train data

0.25 point(s) : generate predictions for train data

0.25 points(s) : generate predictions for test data

_Learning Goal: Perform classification using a text dataset, using sensible preprocessing, tokenization, and feature engineering scheme_
***
#### 3)


Point Value : 1

0.5 point(s) : generate dictionary with document frequency

0.5 point(s) : calculate IDF for each word

_Learning Goal : Use a count vectorization strategy to create a bag of words. Define TF-IDF vectorization and its components._

***
#### 4)


Point Value : 1

0.5 point(s) : Identify "school" as being a very unique word

0.5 point(s) : Identify "school" as important to the second document

_Learning Goal : Use TF-IDF vectorization with multiple documents to identify words that are important to certain documents_

## Network Analysis

#### 1)


Point Value : 1.5

0.5 point(s) : Determine betweenness centrality for each node

0.25 point(s) : Determine the node with greatest betweenness centrality

0.5 point(s) : Determine degree centrality for each node

0.25 point(s) : Determine the node with greatest degree centrality

_Learning Goal : Compare and calculate degree, clonseness, betweenness and eigenvector centrality measures._
***
#### 2)


Point Value : 2

1 point(s) : Correctly explain that a node with higher betweenness centrality is more important for information flow through a network.  

1 point(s) : Correctly explain that the higher degree centrality the means more connections coming into a node. 

_Learning Goal : Interpret characteristics of certain nodes based on their centrality metrics._
***
#### 3)



Point Value : 1.5

1.5 point(s) : Identify 12 communities with clique size of 5

_Learning Goal : Implement k-clique and Girvan-Newman clustering via networkx._

## Perceptron and Multi Layer Perceptron

#### 1)

Learning Goal : Describe inputs and outputs of a perceptron

Point Value : 1

0.5 point(s) : describe inputs as features of observations or the outputs of other activation functions, each assigned a weight

0.5 point(s) : describe outputs as the result of the dot product of the features and the weights associated with them


_Learning Goal : Explain the mechanics of a deep neural network_

***
#### 2)


Point Value : 1.25

0.5 point(s) : Apply dot product to matrix and weights 

0.25 point(s) : Adds the bias term to the dot product

0.5 point(s) : Apply sigmoid function to output of dot product 

_Learning Goal : Compare and contrast the different activation functions_
***


#### 3)



Point Value : 1

0.5 point(s) : Describe the process of applying/multiplying weights to/by inputs at each layer

0.5 point(s) : Explain that the final decision layer of the neural network is used to generate a probability on what class the set of inputs belong to

_Learning Goal : Explain forward propogation in a neural network._
***
#### 4)


Point Value : 1

0.5 point(s) : Describe the process of calculating error
0.5 point(s) : that the magnitude of the error determines by how much the weights need to be updated during the gradient descent of back propogation

_Learning Goal : Explain backward propagation and discuss how it is related to forward propogation._

***
#### 5)


Point Value : 0.75

0.25 point(s) : Explain the problem is a multi-class problem

0.5 point(s) : Explain that a softmax activation function would facilitate a multi-class classifier

_Learning Goal : Compare and contrast the different activation functions. List the different activation functions. Explain the mechanics of a deep neural network._

## Optimization and Regularization of NN

#### 1)


Point Value : 2

1 point(s) : Modify the regularization parameter

0.5 point(s) : Uses L2 regularization

0.5 point(s) : Uses the train and test accuracies to explain that the regularization did prevent overfitting

_Learning Goal : Apply L1, L2, and dropout regularization on a neural network_
***
#### 2)


Point Value : 2

1 point(s) : Explain how regularization can reduce the high variance that neural networks are prone to. 

1 point(s) : Explain that regularization adds penalties to the cost function and prevents any one feature from having too much importance in a model


_Learning Goal : Explain the relationship between bias and variance in neural networks._

***
#### 3)



Point Value : 1

1 point(s) : Explain that L1 can force weights down to zero and effectively "kill" a node

_Learning Goal : Explain how regularization affects the nodes of a neural network_

