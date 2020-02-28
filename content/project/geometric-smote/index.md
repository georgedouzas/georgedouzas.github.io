---
title: Geometric SMOTE
summary: A Python package for flexible and efficient oversampling.
tags:
- Machine Learning
- Imbalanced Learning Problem
- Geometric SMOTE
date: "2019-06-01T00:00:00Z"

# Optional external URL for project (replaces project detail page).
external_link: ""

links:
- icon: github
  icon_pack: fab
  name: Repository
  url: https://github.com/AlgoWit/geometric-smote
url_code: ""
url_pdf: ""
url_slides: ""
url_video: ""

# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
slides: ""
---

## Introduction

Classification of imbalanced datasets is a challenging task for standard machine learning algorithms. Training a classifier on imbalanced data, often results in a low out-of-sample accuracy for the minority classes. To deal with this problem several approaches have been proposed. A general approach, known as oversampling, is the generation of artificial data for the minority classes that are used to enhance the training data. 

SMOTE is the most popular oversampler, while many variants of it have been developed. SMOTE data generation mechanism consists of linerarly interpolating synthetic data between two randomly chosen neighboring minority class instances. A Python implementation of SMOTE and several of its variants is available in the [Imbalanced-Learn](https://imbalanced-learn.org/stable/) library, which is fully compatible with the popular machine learning toolbox [Scikit-Learn](https://scikit-learn.org/stable/). 

Geometric SMOTE, a geometric modification of the SMOTE data generation mechanism, is a state-of-the-art oversampling algorithm that [has been shown](../../publication/gsmote_journal) to outperform other standard oversamplers in a large number of datasets. Geometric SMOTE data generation mechanism consists of generating synthetic data inside a hypersphere that is defined by a randomly chosen minority class instance and one of its neighbors either from the minority or majority class. I have developed a Python implementation of Geometric SMOTE oversampler, called `geometric-smote`, that integrates seamlessly with the Scikit-Learn and Imblanced-Learn ecosystems.

{{< figure src="smote_vs_gsmote.png" title="SMOTE vs Geometric SMOTE data generation mechanisms." width="700px" >}}


## Usage

Detailed documentation that includes installation guidelines, API description and various examples can found [here](https://geometric-smote.readthedocs.io/en/latest/?badge=latest). In what follows, I will describe briefly some aspects of `geometric-smote`'s functionality.

The class that represents the Geometric SMOTE oversampler is called `GeometricSMOTE`. Its API follows closely the API of oversamplers provided by Imbalanced-Learn. In order to show its functionality I will initially generate some binary class imbalanced data, represented by the input matrix `X` and the target vector `y`:

```python
# Imports
from sklearn.datasets import make_classification

# Set random seed
rnd_seed = 40

# Generate imbalanced data
X, y = make_classification(
  n_classes=2,
  weights=[0.9, 0.1],
  random_state=rnd_seed
)
```
The following code snippet prints the main characteristics of the dataset:

```python
# Imports
from collections import Counter

# Define function to extract imbalanced dataset's characteristics
def extract_characteristics(X, y):
  n_samples, n_features = X.shape
  count_y = Counter(y)
  (maj_label, n_samples_maj), (min_label, n_samples_min) = count_y.most_common()
  return n_samples, n_features, maj_label, n_samples_maj, min_label, n_samples_min

# Define function to print imbalanced dataset's characteristics
def print_characteristics(X, y):
  n_samples, n_features, maj_label, n_samples_maj, min_label, n_samples_min = extract_characteristics(X, y)
  print(
    f'Number of samples: {n_samples}',
    f'Number of features: {n_features}',
    f'Majority class label: {maj_label}',
    f'Number of majority class samples: {n_samples_maj}',
    f'Minority class label: {min_label}',
    f'Number of minority class samples: {n_samples_min}',
    f'Imbalance Ratio: {n_samples_maj / n_samples_min:.1f}',
    sep='\n'
  )

# Print imbalanced dataset's characteristics
print_characteristics(X, y)

# Number of samples: 100
# Number of features: 20
# Majority class label: 0
# Number of majority class samples: 89
# Minority class label: 1
# Number of minority class samples: 11
# Imbalance Ratio: 8.1
```

Following the Imbalanced Learn's API, the `fit_resample` method of a `GeometricSMOTE` instace can be used to resample the imbalanced dataset:

```python
# Imports
from gsmote import GeometricSMOTE

# Create GeometricSMOTE instance
gsmote = GeometricSMOTE(random_state=rnd_seed + 5)

# Fit and resample imbalanced data
X_res, y_res = gsmote.fit_resample(X, y)

# Print balanced dataset's characteristics
print_characteristics(X_res, y_res)

# Number of samples: 178
# Number of features: 20
# Majority class label: 0
# Number of majority class samples: 89
# Minority class label: 1
# Number of minority class samples: 89
# Imbalance Ratio: 1.0
```

Therefore the `GeometricSMOTE` instance has generated the apropriate number of minority class instances so that the resampled dataset is perfeclty balanced. As I mentioned above, training a classifier on imbalanced data may result in suboptimal performance on out-of-sample data:

```python
# Imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer
from imblearn.pipeline import Pipeline
from imblearn.metrics import geometric_mean_score

# Define function that calculates out-of-sample score
def calculate_cv_score(oversampler, X, y):
  cv_scores= []
  scoring = make_scorer(geometric_mean_score)
  classifier = DecisionTreeClassifier()
  if oversampler is not None:
    classifier = Pipeline([('ovr', oversampler), ('clf', classifier)])
  for ind in range(100):
    
      classifier = DecisionTreeClassifier(random_state=rnd_seed + 12 * ind)
    
    cv_score = cross_val_score(
      estimator=classifier,
      X=X,
      y=y,
      scoring=scoring,
      cv=StratifiedKFold(shuffle=True, random_state=rnd_seed + 8 * ind)
    ).mean()
    cv_scores.append(cv_score)
  return sum(cv_scores) / len(cv_scores)
# Print cross-validation score when imbalanced data are used

print(f'Cross-validation geometric mean score for imbalanced data: {imbalanced_cross_val_scores.mean():.2f}')

# Cross-validation f1 score: 0.65
```

The correct way to use the resampled data is to integrate the `GeometricSMOTE` oversampler into a pipeline. Then the out-of-sample performance is increased:

```python
# Print cross-validation score when balanced data are used
balanced_cross_val_scores = cross_val_score(
  estimator=make_pipeline(gsmote, classifier),
  X=X,
  y=y,
  scoring=scoring,
  cv=StratifiedKFold()
)
print(f'Cross-validation geometric mean score for balanced data: {balanced_cross_val_scores.mean():.2f}')
```