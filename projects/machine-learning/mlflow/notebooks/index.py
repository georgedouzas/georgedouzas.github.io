# %% [markdown]
# ---
# title: "MLflow"
# description: "An open source platform for the machine learning lifecycle."
# author: "Georgios Douzas"
# date: "2023-06-022"
# categories: [Machine Learning, Open Source, MLOps]
# image: "featured.png"
# jupyter: python3
# draft: true
# ---

# %% [markdown]
"""
![](featured.png)

## Introduction

[MLflow](https://mlflow.org/) is an open-source platform designed to manage the entire machine learning lifecycle. It encompasses
four main functionalities:

- **Tracking**: Monitors experiments, logging parameters and outcomes.  
- **Projects**: Enables the packaging of ML code into a reproducible and shareable format.  
- **Models**: Oversees the management and deployment of models across various ML libraries and serving platforms.  
- **Model Registry**: Central repository that facilitates collaborative management of MLflow Models throughout their
lifecycle, including aspects like versioning and stage transitions.

Notably, MLflow is library-agnostic and supports integration with various ML libraries and programming languages through REST API,
CLI, and additional APIs for Python, R, and Java.

## Tracking

Let's assume we would like to run multiple classification experiments with various classifiers, datasets and random seeds.
"""

# %%
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, cross_validate
import mlflow

rnd_seed = 23


def run_experiment(clf, param_grid, X, y):
    cv = KFold(n_splits=3, shuffle=True, random_state=56 * rnd_seed)
    gscv = GridSearchCV(clf, param_grid=param_grid, cv=cv)
    results = cross_validate(gscv, X, y, cv=cv, scoring=['accuracy', 'f1_micro'])
    return results


clfs = [KNeighborsClassifier(), GradientBoostingClassifier()]
param_grids = [
    {'n_neighbors': [3, 5]},
    {'max_depth': [3, 5], 'n_estimators': [50, 100], 'random_state': [rnd_seed + 10]},
]
data = [load_breast_cancer(return_X_y=True), load_iris(return_X_y=True)]

# %% [markdown]
"""
We can manually track the parameters and results of an experiment.
"""

# %%
results = []
for clf, param_grid in zip(clfs, param_grids):
    for X, y in data:
        results.append(run_experiment(clf, param_grid, X, y))

# %% [markdown]
"""
A simple call to MLflow's `autolog` function can be used instead of manual tracking:
"""

# %%
mlflow.autolog()

for clf, param_grid in zip(clfs, param_grids):
    for X, y in data:
        results.append(run_experiment(clf, param_grid, X, y))
