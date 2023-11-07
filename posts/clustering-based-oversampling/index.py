# %% [markdown]
# ---
# title: Clustering-based oversampling
# description: "Combining clustering and oversampling to increase classification performance."
# author: "Georgios Douzas"
# date: "2022-05-02"
# categories: [Project, Imbalanced Learning]
# image: "featured.png"
# jupyter: python3
# ---

# %% [markdown]
"""
![](featured.png)

## Introduction

SMOTE algorithm and its variants generate synthetic samples along line segments that join minority class instances. SMOTE
addresses only the between-classes imbalance. On the other hand, SMOTE does nothing about areas of the input space that differ
significantly in the density of a particular class, an issue known as within-classes imbalance.

A straightforward approach is to combine oversamplers with clustering algorithms.
[SOMO](https://www.sciencedirect.com/science/article/abs/pii/S0957417417302324) and
[KMeans-SMOTE](https://www.sciencedirect.com/science/article/abs/pii/S0020025518304997) algorithms are specific realizations of
this approach that have been shown to outperform other standard oversamplers in a large number of datasets.

## Implementation

I have developed a Python implementation of the above clustering-based oversampling approach called
[cluster-over-sampling](https://github.com/georgedouzas/cluster-over-sampling), which integrates seamlessly with the
[Scikit-Learn](https://scikit-learn.org/stable/) and [Imbalanced-Learn](https://imbalanced-learn.org/stable/) ecosystems. You can
check the [documentation](https://github.com/georgedouzas/cluster-over-sampling) for more details on installation and the API.

## Functionality

Let's first generate a binary class imbalanced dataset, represented by the input matrix `X` and the target vector `y`. Using a
high value for the `flip_y` parameter, we ensure that the data are noisy thus the clustering of the input space will help the
oversampling process:
"""

# %%
# Imports
from sklearn.datasets import make_classification

# Set random seed
rnd_seed = 4

# Generate imbalanced data
X, y = make_classification(
    n_samples=500,
    n_classes=2,
    weights=[0.9, 0.1],
    random_state=rnd_seed,
    n_informative=3,
    class_sep=1.0,
    n_features=10,
    flip_y=0.3,
)

# %% [markdown]
"""
The function `print_characteristics` extracts and prints the main characteristics of a binary class dataset. Specifically, it
prints the number of samples, the number of features, the labels, and the number of samples for the majority and minority classes
as well as the Imbalance Ratio, defined as the ratio between the number of instances of the majority and minority classes.
"""

# %%
# Imports
from collections import Counter


# Define function to print dataset's characteristics
def print_characteristics(X, y):
    n_samples, n_features = X.shape
    count_y = Counter(y)
    (maj_label, n_samples_maj), (min_label, n_samples_min) = count_y.most_common()
    ir = n_samples_maj / n_samples_min
    print(
        f'Number of samples: {n_samples}',
        f'Number of features: {n_features}',
        f'Majority class label: {maj_label}',
        f'Number of majority class samples: {n_samples_maj}',
        f'Minority class label: {min_label}',
        f'Number of minority class samples: {n_samples_min}',
        f'Imbalance Ratio: {ir:.1f}',
        sep='\n',
    )


# %% [markdown]
"""
I use the above function to print the main characteristics of the generated imbalanced dataset:
"""

# %%
print_characteristics(X, y)

# %% [markdown]
"""
I will combine the `SMOTE` oversampler and the `KMeans` clusterer to rebalance the above dataset. The combined
clusterer-oversampler can be constructed by importing the `SMOTE` oversampler from Imbalanced-Learn, `KMeans` from Scikit-Learn,
and `ClusterOverSampler` from `cluster-over-sampling`. Then following the Imbalanced Learn API, we can use the `fit_resample`
method to resample the imbalanced dataset:
"""

# %%
# Imports
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from clover.over_sampling import ClusterOverSampler

# Create KMeans-SMOTE instance
smote = SMOTE(random_state=rnd_seed + 1)
kmeans = KMeans(n_clusters=10, random_state=rnd_seed + 3, n_init=50)
kmeans_smote = ClusterOverSampler(oversampler=smote, clusterer=kmeans)

# Fit and resample imbalanced data
X_res, y_res = kmeans_smote.fit_resample(X, y)

# %% [markdown]
"""
Again we can print the main characteristics of the rebalanced dataset:
"""

# %%
print_characteristics(X_res, y_res)

# %% [markdown]
"""
The default behavior is to generate the appropriate number of minority class samples so that the resampled dataset is perfectly
balanced (although this sometimes may result in an approximately balanced dataset). Also, cluster-over-sampling provides for
convenience, the clustering-based oversamplers [SOMO](https://www.sciencedirect.com/science/article/abs/pii/S0957417417302324) and
[KMeans-SMOTE](https://www.sciencedirect.com/science/article/abs/pii/S0020025518304997), as well as
[G-SOMO](https://www.sciencedirect.com/science/article/abs/pii/S095741742100662X) that uses [Geometric
SMOTE](../../publication/gsmote_journal) as the oversampler in place of SMOTE.

As mentioned above, training a classifier on imbalanced data may result in suboptimal performance on out-of-sample data. The
function `calculate_cv_scores` calculates the average 5-fold cross-validation F and accuracy scores across 5 runs
of a `RandomForestClassifier` that is optionally combined with an oversampler through a pipeline:
"""

# %%
# Imports
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from imblearn.pipeline import make_pipeline


# Define function that calculates out-of-sample scores
def calculate_cv_scores(oversampler, X, y):
    cv_scores = []
    scorers = {
        'f_score': make_scorer(f1_score),
        'accuracy': make_scorer(accuracy_score),
    }
    n_runs = 5
    for ind in range(n_runs):
        rnd_seed = 8 * ind
        classifier = RandomForestClassifier(random_state=rnd_seed)
        if oversampler is not None:
            classifier = make_pipeline(
                oversampler.set_params(random_state=rnd_seed + 5), classifier
            )
        scores = cross_validate(
            estimator=classifier,
            X=X,
            y=y,
            scoring=scorers,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=rnd_seed + 6),
        )
        cv_scores.append([scores[f'test_{scorer}'].mean() for scorer in scorers])
    return np.mean(cv_scores, axis=0)


# %% [markdown]
"""
Using the above function, we can calculate the out-of-sample performance when no oversampling is applied, as well as when SMOTE
and DBSCAN-SMOTE are used as oversamplers:
"""

# %%
# Imports
from sklearn.cluster import DBSCAN
from clover.over_sampling import ClusterOverSampler

# Calculate cross-validation scores
mapping = {
    'No oversampling': None,
    'SMOTE': SMOTE(),
    'DBSCAN-SMOTE': ClusterOverSampler(oversampler=SMOTE(), clusterer=DBSCAN()),
}
cv_scores = {}
for name, oversampler in mapping.items():
    cv_scores[name] = calculate_cv_scores(oversampler, X, y)
cv_scores = pd.DataFrame(cv_scores, index=['F-score', 'Accuracy'])
cv_scores

# %% [markdown]
"""
Notice that using accuracy as an evaluation metric is not a good choice when the data is imbalanced. For example, a
trivial classifier that always predicts the majority class would still have an accuracy equal to 0.90, even though all the
minority class instances are misclassified. On the other hand, the F-score is an appropriate evaluation metric for
imbalanced data since it considers the accuracies per class.
"""
