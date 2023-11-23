# %% [markdown]
# ---
# title: cluster-over-sampling
# description: "A general interface for clustering based over-sampling algorithms."
# author: "Georgios Douzas"
# date: "2022-05-02"
# categories: [Machine Learning, Open Source, Imbalanced Data]
# image: "featured.png"
# jupyter: python3
# ---

# %% [markdown]
"""
![](featured.png)

SMOTE algorithm and its variants generate synthetic samples along line segments that join minority class instances. SMOTE
addresses only the between-classes imbalance. On the other hand, SMOTE does nothing about areas of the input space that differ
significantly in the density of a particular class, an issue known as within-classes imbalance. An approach that deals with this
issue is to use a clusterer to select sparse areas of the input space and then apply oversampling to them.  I have developed a
Python implementation of the above clustering-based oversampling approach called
[cluster-over-sampling](https://github.com/georgedouzas/cluster-over-sampling), which integrates seamlessly with the
[Scikit-Learn](https://scikit-learn.org/stable/) and [Imbalanced-Learn](https://imbalanced-learn.org/stable/) ecosystems. To
present the API, let's first load some data:
"""

# %%
# Imports
from sklearn.datasets import load_breast_cancer

# Load data
X, y = load_breast_cancer(return_X_y=True)

# %% [markdown]
"""
The data are clearly imbalanced:
"""

# %%
# Imports
from collections import Counter

# Classes distribution
counter = Counter(y)
print(
    f"Number of majority class samples: {counter[1]}.",
    f"Number of minority class samples: {counter[0]}.",
    sep="\n",
)

# %% [markdown]
"""
I will use `KMeans` and `SMOTE` to create a clustering-based oversampler but any combination would work:
"""

# %%
# Imports
from sklearn.datasets import load_breast_cancer
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from clover.over_sampling import ClusterOverSampler

# Create KMeans-SMOTE instance
rnd_seed = 14
smote = SMOTE(random_state=rnd_seed + 1)
kmeans = KMeans(n_clusters=10, random_state=rnd_seed + 3, n_init=50)
kmeans_smote = ClusterOverSampler(oversampler=smote, clusterer=kmeans)

# %% [markdown]
"""
Now we can use the `fit` method to calculate statistics of the oversampling process:
"""

# %%
kmeans_smote.fit(X, y)
print(
    f"Number of generated minority class instances to rebalance dataset: {kmeans_smote.sampling_strategy_[0]}."
)

# %% [markdown]
"""
Or use the `fit_resample` method to get the resampled data:
"""

# %%
_, y_res = kmeans_smote.fit_resample(X, y)
counter = Counter(y_res)
print(
    f"Number of majority class samples: {counter[1]}.",
    f"Number of minority class samples: {counter[0]}.",
    sep="\n",
)

# %% [markdown]
"""
The clustering-based oversamplers can be used in machine learning pipelines:
"""

# %%
# Imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from imblearn.pipeline import make_pipeline

# Cross validation score
classifier = RandomForestClassifier(random_state=rnd_seed)
classifier = make_pipeline(kmeans_smote, classifier)
score = cross_val_score(estimator=classifier, X=X, y=y, scoring="f1").mean()
print(f"The cross-validation F-score is {score}.")

# %% [markdown]
"""
For more details and examples you can check the [documentation](https://georgedouzas.github.io/cluster-over-sampling/).
"""
