# Implementation Overview

This page provides a high-level overview of the implementation.
In essence, this overview is a combination of three things:

1. Section 8.1.1 Regression Trees and 8.1.2 Classification trees (James et al., [2021](https://doi.org/10.1007/978-1-0716-1418-1)).
2. The SIRUS algorithm description (Bénard et al., [2021](http://proceedings.mlr.press/v130/benard21a.html)).
3. Some implementation details, as obtained by trial and error and by correspondence with Clement Bénard, which were missing from aforementioned sources.

## Fit Stabilized Trees

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

## Rule Generation

After creating many trees, the SIRUS algorithm converts these trees to rules.
One of the first of such rule-based models was the RuleFit algorithm (Friedman & Popescu, [2008](http://www.jstor.org/stable/30245114)).
The idea behind these models is that any tree can be expressed as a set of rules.
For example, say we have the following tree with nodes ``n_1, n_2, ..., n_5``.

```@setup tree
using CairoMakie

empty_theme = Theme(
    Axis = (
        backgroundcolor = :transparent,
        leftspinevisible = false,
        rightspinevisible = false,
        bottomspinevisible = false,
        topspinevisible = false,
        xticklabelsvisible = false,
        yticklabelsvisible = false,
        xgridcolor = :transparent,
        ygridcolor = :transparent,
        xminorticksvisible = false,
        yminorticksvisible = false,
        xticksvisible = false,
        yticksvisible = false,
        xautolimitmargin = (0.0,0.0),
        yautolimitmargin = (0.0,0.0),
    )
)

function plot_tree()
    with_theme(empty_theme) do
        fig = Figure()
        ax = Axis(fig[1, 1])
        linesopts = (
            color = :black,
            space = :relative,
        )
        scatteropts = (
            color = :white,
            markersize = 100,
            strokewidth = 2,
            space = :relative,
            transparency = true
        )
        textopts = (
            space = :relative,
            fontsize = 30,
            justification = :center,
            align = (:center, :center)
        )

        lines!(ax, [0.5, 0.3], [0.9, 0.5]; linesopts...)
        lines!(ax, [0.5, 0.7], [0.9, 0.5]; linesopts...)

        lines!(ax, [0.3, 0.12], [0.5, 0.1]; linesopts...)
        lines!(ax, [0.3, 0.48], [0.5, 0.1]; linesopts...)

        scatter!(ax, 0.5, 0.9; scatteropts...)
        text!(ax, 0.5, 0.9; text=L"n_1", textopts...)
        scatter!(ax, 0.3, 0.5; scatteropts...)
        text!(ax, 0.3, 0.5; text=L"n_2", textopts...)
        scatter!(ax, 0.7, 0.5; scatteropts...)
        text!(ax, 0.7, 0.5; text=L"n_3", textopts...)

        scatter!(ax, 0.12, 0.1; scatteropts...)
        text!(ax, 0.12, 0.1; text=L"n_4", textopts...)
        scatter!(ax, 0.48, 0.1; scatteropts...)
        text!(ax, 0.48, 0.1; text=L"n_5", textopts...)

        text!(ax, 0.35, 0.75; text=L"x_1 < 3", textopts...)
        text!(ax, 0.65, 0.75; text=L"x_1 \geq 3", textopts...)

        text!(ax, 0.16, 0.33; text=L"x_2 < 5", textopts...)
        text!(ax, 0.45, 0.33; text=L"x_2 \geq 5", textopts...)

        hidedecorations!(ax)
        return fig
    end
end
```

```@example tree
plot_tree() # hide
```

and say that this tree was generated from a tree fitting procedure as described above.
From this representation, we can see that node ``n_1`` splits the feature ``x_1`` on 3.
If ``x_1 < 3``, then the prediction will go to ``n_2`` and if ``x \geq 3``, then the prediction will take the content of ``n_3``.
In ``n_2``, the prediction will be made based on ``n_4`` or ``n_5`` depending on whether feature ``x_2`` is smaller than or greater or equal to 5.

To convert such a tree to rules, note that each path to a leaf can be converted to one rule.
For example, the path to ``n_3`` can be converted to

```math
\text{if } x_1 \geq 3, \text{ then } A \text{ else } B,
```

where ``A`` considers all points that satisfy the rule constraints represented by the path to ``n_3`` (``x_1 \geq 3``) and ``B`` considers all points that do not satisfy the rule constraints (``x_1 < 3``).
Similarly, the path to ``n_4`` can be converted to

```math
\text{if } x_1 < 3 \: \& \: x_2 < 5, \text{ then } C \text{ else } D,
```

where ``C`` considers all the points that satisfy the rule constraints and ``D`` considers all the points that do not satisfy the rule constraints.

Unlike regular decision trees, rule-based models are typically restricted to a depth of 2 for reasons that will be explained later.
For now, say we have a large number of trees, typically around 1500, with a depth of at most 2, then we can estimate that the number of generated rules will be at most 6 * 1500 = 9000.
I'm taking 6 here since a tree of depth 2 can have at most 7 nodes, of which all but the root are converted to a separate rule.
In other words, "both internal and external nodes are extracted from the trees of the random forest to generate a large collection of rules, typically ``10^4``" (Bénard et al., [2021](http://proceedings.mlr.press/v130/benard21a.html), Section _Rule generation_).

Obviously, thousands of rules are not interpretable.
Just like random forests, each prediction can be traced back to exactly how it was made.
However, thousands of additions and multiplications are hard to interpret and explain.
That's why visualizations based on SHAP are often used (e.g., Molnar, [2023](https://christophm.github.io/interpretable-ml-book/)).
Still, these visualizations simplify the model such that feature importances becomes explainable, but they do not fully explain how predictions are made.
The SIRUS algorithm solves that by simplifying the model itself instead of only simplifying the model explanation.

## Rule Selection and Post-Treatment

The aim of rule selection is to remove as many rules as possible without affecting the predictive performance too much.
This is similar to pruning in artificial neural networks, where the least important neurons are removed.
This is done via two ways.
Firstly, by pruning the many identical rules in the different forests, which are identical thanks to tree stabilization done when building the trees.
Secondly, by pruning rules which are a linear combination of other rules.

The first step requires setting a hyperparameter ``p_0`` in the original algorithm.
This hyperparameter specifies a threshold which is then used to remove rules with a occurence frequency below ``p_0``.
The implementation in this Julia package ignores this step because it is superseeded by the second step and because it is quite difficult to pick the right ``p_0``.
In other words, this step is ignored because it seems like a premature optimization (but I'm happy to be proven wrong of course).

The second step is more important and more involved.
As said before, the second step is to remove the least important linear combinations of other rules.
An example of this is shown in the original paper (Bénard et al., [2021](http://proceedings.mlr.press/v130/benard21a.html), Table 3 (the second) in Section 4 _Post-treatment Illustration_ of the Supplementary PDF).

The implementation for this can be done by converting the training data to a feature space in which each rule becomes a binary feature indicating whether the data point satisfies the constraint or not.
This is quite computationally intensive since there is a lot of duplication in the data and it doesn't guarantee that all cases of duplication will be found since some may not be in the training set.
Luckily, D.W. on StackExchange (https://cs.stackexchange.com/questions/152803) has provided a solution, which I will repeat here.
The idea is to remove each rule ``r`` when it is linearly dependent on the preceding rules.

To do this, observe that a rule of the form ``A \: \& \: B`` can only depend on rules that use some combination of ``A``, ``!A``, ``B``, and/or ``!B``.
This works by iteratively calculating the rank and seeing whether the rank increases.

We can assume that we are limited to a set of rules where either `A & B`, `A & !B`, `!A & B`, `!A & !B`, `A`, `!A`, `B`, `!B` or `True`.
This last case is not a valid rule in this algorithm, so that will not happen.
Now, given `A` and `B`, we can create a binary matrix with a row for `A & B`, `A & !B`, `!A & B`, `!A & !B`.
Next, generate one column containing `true`s and one column for each rule in `rules`.
In each column, answer whether the rule holds for some point that satisifies the conditional.
Next, calculate the rank and see whether the rank increases when adding additional rules.
