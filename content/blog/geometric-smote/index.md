---
title: Geometric SMOTE algorithm
date: "2022-05-01"
description: "A practical introduction to the Geometric SMOTE algorithm."
---

![Geometric SMOTE algorithm](featured.png "Geometric SMOTE algorithm")

## Introduction

Classification of imbalanced datasets is a challenging task for standard machine learning algorithms. Training a classifier on imbalanced data, often results in a low out-of-sample accuracy for the minority classes. To deal with this problem several approaches have been proposed. A general approach, known as oversampling, is the generation of artificial data for the minority classes that are used to enhance the training data.

*SMOTE* is the most popular oversampling algorithm, while many variants of it have been developed. SMOTE generates synthetic data between the line segment that connects two randomly chosen neighboring minority class instances. On the other hand, Geometric SMOTE expands the data generation area by generating synthetic data inside a hypersphere that is defined by a randomly chosen minority class instance and one of its neighbors either from the minority or majority class. [*Geometric SMOTE*](https://www.sciencedirect.com/science/article/abs/pii/S0020025519305353) has been shown to outperform other standard oversamplers in a large number of datasets. The following figure illustrates the difference between the two data generation mechanisms:

![SMOTE vs Geometric SMOTE](smote_vs_gsmote.png "SMOTE vs Geometric SMOTE")

A Python implementation of *SMOTE* and several of its variants is available in the [*Imbalanced-Learn*](https://imbalanced-learn.org/stable/) library, which is fully compatible with the popular machine learning toolbox [*Scikit-Learn*](https://scikit-learn.org/stable/). I have developed a *Python* implementation of the *Geometric SMOTE* oversampler, called [*geometric-smote*](https://github.com/georgedouzas/geometric-smote), that integrates seamlessly with the *Scikit-Learn* and *Imbalanced-Learn* ecosystems.


## Installation

The easiest way to install the *geometric-smote* package, assuming you have already *Python 3* and *pip* installed as well as you have optionally activated a *Python virtual environment*, is to open a shell and run the following command:

```
pip install geometric-smote
```

This will install the latest version and its dependencies.

## Documentation

Detailed documentation that includes installation guidelines, API description and various examples can found [here](https://geometric-smote.readthedocs.io/en/latest/?badge=latest). 

## Functionality

In what follows, I will describe briefly some aspects of *geometric-smote* functionality. The class that represents the Geometric SMOTE oversampler is called `GeometricSMOTE`. Its API follows closely the API of oversamplers provided by *Imbalanced-Learn*. 

#### Resampling an imbalanced dataset

Let's generate a binary class imbalanced dataset, represented by the input matrix `X` and the target vector `y`:

```python
# Imports
from sklearn.datasets import make_classification

# Set random seed
rnd_seed = 43

# Generate imbalanced data
X, y = make_classification(
  n_samples=100,
  n_classes=2,
  weights=[0.9, 0.1],
  random_state=rnd_seed
)
```

The following functions extract and print the main characteristics of a binary class dataset. Specifically, the `extract_characteristics` function returns the number of samples, the number of features,  the labels and the number of samples for the majority and minority classes as well as the Imbalance Ratio defined as the ratio between the number of samples of the majority and minority classes, while the `print_characteristics` function prints them in an appropriate format:

```python
# Imports
from collections import Counter

# Define function to extract dataset's characteristics
def extract_characteristics(X, y):
  n_samples, n_features = X.shape
  count_y = Counter(y)
  (maj_label, n_samples_maj), (min_label, n_samples_min) = count_y.most_common()
  ir = n_samples_maj / n_samples_min
  return n_samples, n_features, maj_label, n_samples_maj, min_label, n_samples_min, ir

# Define function to print dataset's characteristics
def print_characteristics(X, y):
  n_samples, n_features, maj_label, n_samples_maj, min_label, n_samples_min, ir = extract_characteristics(X, y)
  print(
    f'Number of samples: {n_samples}',
    f'Number of features: {n_features}',
    f'Majority class label: {maj_label}',
    f'Number of majority class samples: {n_samples_maj}',
    f'Minority class label: {min_label}',
    f'Number of minority class samples: {n_samples_min}',
    f'Imbalance Ratio: {ir:.1f}',
    sep='\n'
  )
```

I use the above function to print the main characteristics of the generated imbalanced dataset:

```python
# Print imbalanced dataset's characteristics
print_characteristics(X, y)

##########
# Output #
##########

# Number of samples: 100
# Number of features: 20
# Majority class label: 0
# Number of majority class samples: 90
# Minority class label: 1
# Number of minority class samples: 10
# Imbalance Ratio: 9.0
```

Following the *Imbalanced-Learn* API, the `fit_resample` method of a `GeometricSMOTE` instance can be used to resample the imbalanced dataset:

```python
# Imports
from gsmote import GeometricSMOTE

# Create GeometricSMOTE instance
geometric_smote = GeometricSMOTE(random_state=rnd_seed + 5)

# Fit and resample imbalanced data
X_res, y_res = geometric_smote.fit_resample(X, y)
```

Again we can print the main characteristics of the rebalanced dataset:

```python
# Print balanced dataset's characteristics
print_characteristics(X_res, y_res)

##########
# Output #
##########

# Number of samples: 180
# Number of features: 20
# Majority class label: 0
# Number of majority class samples: 90
# Minority class label: 1
# Number of minority class samples: 90
# Imbalance Ratio: 1.0
```

As expected, the default behavior of the `GeometricSMOTE` instance is to generate the appropriate number of minority class instances so that the resampled dataset is perfectly balanced. 

#### Performance on out-of-sample data

As I mentioned above, training a classifier on imbalanced data may result in suboptimal performance on out-of-sample data. The function `calculate_cv_scores` calculates the average 10-fold cross-validation geometric mean and accuracy scores across 100 runs of a decision tree classifier that is optionally combined to an oversampler through a pipeline:

```python
# Imports
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, SCORERS
from imblearn.pipeline import make_pipeline
from imblearn.metrics import geometric_mean_score

# Append geometric mean score
SCORERS['g_mean'] = make_scorer(geometric_mean_score)

# Define function that calculates out-of-sample scores
def calculate_cv_scores(oversampler, X, y):
  mean_cv_scores= []
  scoring = ['g_mean', 'accuracy']
  n_runs = 100
  for ind in range(n_runs):
    rnd_seed = 10 * ind
    classifier = DecisionTreeClassifier(random_state=rnd_seed)
    if oversampler is not None:
      classifier = make_pipeline(
        oversampler.set_params(random_state=rnd_seed + 4), 
        classifier
      )
    cv_scores = cross_validate(
      estimator=classifier,
      X=X,
      y=y,
      scoring=scoring,
      cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=rnd_seed + 6)
    )
    cv_scores = [cv_scores[f'test_{scorer}'].mean() for scorer in scoring]
    mean_cv_scores.append(cv_scores)
  return cv_scores
```

Using the above function we can calculate the out-of-sample performance when no oversampling is applied as well as when *SMOTE* and *Geometric SMOTE* are used as oversamplers:

```python
# Imports
from imblearn.over_sampling import SMOTE

# Calculate cross-validation scores
mapping = {'No oversampling': None, 'SMOTE': SMOTE(), 'Geometric SMOTE': GeometricSMOTE()}
cv_scores = {}
for name, oversampler in mapping.items():
  cv_scores[name] = calculate_cv_scores(oversampler, X, y)
```

Printing a table of the scores, we see that *Geometric SMOTE* outperforms the other methods when geometric mean score is used as an evaluation metric, while the highest accuracy is achieved when no oversampling is applied:

```python
cv_scores = pd.DataFrame(cv_scores, index = ['Geometric Mean', 'Accuracy'])
print(cv_scores)

##########
# Output #
##########

#                 No oversampling     SMOTE  Geometric SMOTE
# Geometric Mean         0.782843  0.582843         0.841616
# Accuracy               0.950000  0.920000         0.870000
```

Notice that using the accuracy as an evaluation metric is not considered a good choice when the data is imbalanced. For example, a trivial classifier that always predicts the majority class would still have an accuracy equal to 0.90, even though all the minority class instances are misclassified. On the other hand, geometric mean score is an appropriate evaluation metric for imbalanced data since it equally weighs the accuracies per class. 

For more details you can look at the *geometric-smote* [documentation](https://geometric-smote.readthedocs.io/en/latest/?badge=latest). The [documentation](https://imbalanced-learn.readthedocs.io/en/stable/) of the *Imbalanced-Learn* project provides also various examples and an introduction to the imbalanced learning problem.

