# %% [markdown]
# ---
# title: "Geometric SMOTE algorithm"
# description: "Extending SMOTE's data generation mechanism."
# author: "Georgios Douzas"
# date: "2022-05-01"
# categories: [Machine Learning, Open Source, Imbalanced Data]
# image: "featured.png"
# jupyter: python3
# ---

# %% [markdown]
"""
The [SMOTE](https://arxiv.org/pdf/1106.1813.pdf) algorithm is the most popular oversampler, with many proposed variants. On the
other hand, [Geometric SMOTE](https://www.sciencedirect.com/science/article/abs/pii/S0020025519305353) is not another member of
the SMOTE's family since it expands the data generation area and does not use linear interpolation for new samples. You can check
the following figure for a visual representation of the their difference:

![](smote_vs_gsmote.png)

I have developed a Python implementation of the Geometric SMOTE oversampler called
[geometric-smote](https://github.com/georgedouzas/geometric-smote), which integrates with the
[Scikit-Learn](https://scikit-learn.org/stable/) and [Imbalanced-Learn](https://imbalanced-learn.org/stable/) ecosystems. To run a
comparison experiment, let's first create various imbalanced binary datasets with different characteristics:
"""

# %%

# Imports
from sklearn.datasets import make_classification
from sklearn.model_selection import ParameterGrid

# Set random seed
rnd_seed = 43

# Generate imbalanced datasets
datasets = []
datasets_params = ParameterGrid(
    {"weights": [[0.8, 0.2], [0.9, 0.1]], "class_sep": [0.01, 0.1]}
)
for data_params in datasets_params:
    datasets.append(
        make_classification(
            random_state=rnd_seed,
            n_informative=10,
            n_samples=2000,
            n_classes=2,
            **data_params,
        )
    )

# %% [markdown]
"""
We will also create pipelines of various oversamplers, classifiers and their hyperparameters:
"""

# %%
# Imports
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from gsmote import GeometricSMOTE

# Pipelines
classifiers = [LogisticRegression(), KNeighborsClassifier()]
oversamplers = [None, RandomOverSampler(), SMOTE(), GeometricSMOTE()]
pipelines = []
oversamplers_param_grids = {
    "SMOTE": {
        "smote__k_neighbors": [
            NearestNeighbors(n_neighbors=2),
            NearestNeighbors(n_neighbors=3),
        ]
    },
    "GeometricSMOTE": {
        "geometricsmote__k_neighbors": [2, 3],
        "geometricsmote__deformation_factor": [0.0, 0.25, 0.5, 0.75, 1.0],
    },
}
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=rnd_seed + 5)
for classifier in classifiers:
    for oversampler in oversamplers:
        oversampler_name = (
            oversampler.__class__.__name__ if oversampler is not None else None
        )
        param_grid = oversamplers_param_grids.get(oversampler_name, {})
        estimator = (
            make_pipeline(oversampler, classifier)
            if oversampler is not None
            else make_pipeline(classifier)
        )
        pipelines.append(GridSearchCV(estimator, param_grid, cv=cv, scoring="f1"))

# %% [markdown]
"""
Finally, we will calculate the [nested cross-validation
scores](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html) of the above
pipelines using F-score as evaluation metric:
"""

# %%
#| output: false
n_runs = 3
cv_scores = []
for run_id in range(n_runs):
    for dataset_id, (X, y) in enumerate(datasets):
        for pipeline_id, pipeline in enumerate(pipelines):
            for param in pipeline.get_params():
                if param.endswith("__n_jobs") and param != "estimator__smote__n_jobs":
                    pipeline.set_params(**{param: -1})
                if param.endswith("__random_state"):
                    pipeline.set_params(
                        **{
                            param: rnd_seed
                            * (run_id + 1)
                            * (dataset_id + 1)
                            * (pipeline_id + 1)
                        }
                    )
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=10 * run_id)
            scores = cross_val_score(
                estimator=pipeline,
                X=X,
                y=y,
                scoring="f1",
                cv=cv,
            )
            print(f"Run: {run_id} | Dataset: {dataset_id} | Pipeline: {pipeline_id}")
            pipeline_name = '-'.join([estimator.__class__.__name__ for _, estimator in pipeline.get_params()['estimator'].get_params()['steps']])
            cv_scores.append((run_id, dataset_id, pipeline_name, scores.mean()))

# %% [markdown]
"""
Let's see the final results of the experiment:
"""

# %%
cv_scores = (
    pd.DataFrame(cv_scores, columns=["Run", "Dataset", "Pipeline", "Score"])
    .groupby(["Dataset", "Pipeline"])["Score"]
    .mean()
    .reset_index()
)
cv_scores

# %% [markdown]
"""
The next table shows the pipeline with the highest F-score per dataset:
"""

# %%

cv_scores_best = cv_scores.loc[cv_scores.groupby("Dataset")["Score"].idxmax()]
cv_scores_best

# %% [markdown]
"""
Therefore, Geometric SMOTE outperforms the other methods in all datasets when the F-score is used as an evaluation metric.
"""
