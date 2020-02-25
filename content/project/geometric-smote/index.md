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

SMOTE is the most popular oversampler, while many variants of it have been proposed. Geometric SMOTE, a geometric modification of the SMOTE data generation mechanism, is a state-of-the-art oversampling algorithm that [has been shown](../../publication/gsmote_journal) to outperform other standard oversamplers in a large number of datasets.

A Python implementation of SMOTE and several of its variants is available in the [Imbalanced-Learn](https://imbalanced-learn.org/stable/) library, which is fully compatible with the popular machine learning toolbox [Scikit-Learn](https://scikit-learn.org/stable/).

 I have developed a Python implementation of Geometric SMOTE oversampler, called `geometric-smote`, that integrates seamlessly with the Scikit-Learn and Imblanced-Learn ecosystems.

## Usage

Detailed documentation that includes installation guidelines, API description and various examples can found [here](https://geometric-smote.readthedocs.io/en/latest/?badge=latest). In what follows, I will describe briefly some aspects of `geometric-smote`'s functionality.

The class that represents the Geometric SMOTE oversampler is called `GeometricSMOTE`. Its API follows closely the API of oversamplers provided by Imbalanced-Learn. In order to show its functionality I will initially generate some binary class imbalanced data, represented by the input matrix `X` and the target vector `y`:

```python
# Import scikit-learn
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
The following snippet prints the main characteristics of the dataset:

```python
# Import Counter
from collections import Counter

# Print dataset's characteristics
print(
  'Number of samples'
)
```
