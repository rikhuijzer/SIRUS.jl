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

What is different from vanilla random forest implementations, is that the SIRUS algorithm calculates the cutpoints before splitting.
These cutpoints are calculated over the whole dataset and then the location of the splits are restricted to these pre-determined cutpoints.
In other words, the location of the splits is only allowed to be on one of the pre-determined cutpoints.
Regarding the cutpoints calculation, the cutpoints are calculated by taking `q`-empirical quantiles.
Simply put, taking `q`-empirical quantiles means determining `q` quantiles (cutpoints) which divide the dataset in nearly equal sizes.
The _empirical_ part denotes that we determine the quantiles for data instead of a probability distribution.
