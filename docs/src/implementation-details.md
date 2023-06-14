# Implementation details

This page describes an high-level overview of the implementation.
In essence, this overview is a combination of three things:

1. Section 8.1.1 Regression Trees and 8.1.2 Classification trees (James et al., [2021](https://doi.org/10.1007/978-1-0716-1418-1)).
2. The SIRUS algorithm description (Bénard et al., [2021](http://proceedings.mlr.press/v130/benard21a.html)).
3. Some implementation details which were missing from aforementioned sources.

## Fit stabilized trees

The tree fitting procedure is very similar to the algorithm explanation in James et al. ([2021](https://doi.org/10.1007/978-1-0716-1418-1)).
In summary:

First, fit a large number of trees where for each tree a subset is taken from the observations and the features.
These subsets make the trees less correlated and it has been empirically shown that this improves predictive performance.
Specifically, the subsets are as follows:

- For the subset of observations, take `partial_sampling` (default: 0.7) * `n` random observations from the original dataset.
    Sample these observations with replacement.
- For the subset of features, take `sqrt(p)` random features from the original dataset, where `p` is the number of features.
    Samples these features without replacement.
    Note that the subset of features is chosen at each split of the tree and not only once for each tree.
    If you choose the subset only at the start of the tree building, then an important feature might not end up in the tree at all, which results in poor predictive performance.
    So, chosing this at each split is the best of both worlds since it (1) avoids that each tree splits the root node on the same feature and (2) does still allow the important features to all be used inside the tree.

Before continuing with the algorithm description, we need a small digression on the splitpoints that the algorithm uses.
What is different in the SIRUS algorithm compared to vanilla random forest implementations, is that the SIRUS algorithm calculates the splitpoints before splitting.
These splitpoints are calculated over the whole dataset and then the location of the splits are restricted to these pre-determined splitpoints.
In other words, the location of the splits is only allowed to be on one of the pre-determined splitpoints.
Regarding the splitpoints calculation, the splitpoints are calculated by taking `q`-empirical quantiles.
Simply put, taking `q`-empirical quantiles means determining `q` quantiles (splitpoints) which divide the dataset in nearly equal sizes.
The _empirical_ part denotes that we determine the quantiles for data instead of a probability distribution.

On this subset, then fit a tree.
For both trees, we apply the _top-down_, _greedy_ approach of _recursive binary splitting_, where each split aims to find the best split point.
Finding the best splitpoint means looping through each possible splitpoint from the aforementioned set of pre-determined splitpoints and for each splitpoint determine two half-planes (or regions).
In the left half-plane, take all the points in the feature under consideration which are lower than the splitpoint, that is,
`` R_1 = \{ X \: | \: X_j < s \: \} ``.
In the right half-plane, take all the points in the feature under consideration which are higher or equal than the splitpoint, that is,
`` R_2 = \{ X \: | \: X_j \geq s \: \} ``.
Then for each of this combination of two half-planes, find the best splitpoint.
Finding the best splitpoint boils down to find the split which "summarizes" the data in the best way.
For regression, the best split point is found by finding the splitpoint for which we lose the least information when taking the average of ``R_1`` and ``R_2``.
More formally, the split is found by minimizing the Residual Sum of Squares (RSS):

```math
\sum_{x_i \in R_1} (y_i - \hat{y}_{R_1})^2 + \sum_{x_i \in R_2} (y_i - \hat{y}_{R_2})^2,
```

where ``\hat{y}_{R_1}`` and ``\hat{y}_{R_2}`` denote the mean response for the training observations in respectively ``R_1`` and ``R_2``.
For classification, the best split point is found by determining the classes beforehand and then using these to calculate the Gini index.
The Gini index is needed because classification deals with an unordered set of classes.
The Gini index is a way to determine the most informative splitpoint via _node purity_ and defined as:

```math
1 - \sum_{\text{class} \in \text{classes}} p_{\text{class}}^2,
```

where ``p_\text{classes}`` denotes the fraction (proportion) of items from the current region that are from `class`.
Note that this equation is optimized for computational efficiency.
For the full derivation from the original equation, see _Gini impurity_ at [Wikipedia](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity).
