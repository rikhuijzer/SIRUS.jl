# StableTrees

This package implements the **S**table and **I**nterpretable **RU**le **S**ets (SIRUS) for classification and regression problems (Bénard et al., [2021](https://doi.org/10.1214/20-EJS1792)).

The trees are stabilized by first defining an empirical CART-splitting criterion.
In the context of binary classification, "maximizing the so-called empirical CART-splitting criterion is equivalent to maximizing the criterion on Gini impurity".
Gini impurity is defined as (James et al., [2014](https://doi.org/10.1007/978-1-0716-1418-1); Eq. 8.6):

```math
G = \sum_{k=1}^K \overline{p}_{mk} (1 - \overline{p}_{mk}),
```
where

-  $\overline{p}_{mk}$ represents the proportion of training observations in the $m$-th region that are from the $k$-th class.

Specifically, in Equation 4.1 the empirical CART-splitting criterion is defined as

```math
\newcommand\identity{1\kern-0.25em\text{l}}

\begin{aligned}
L_n(H, \hat{q}_{n,r}^{(j)}) = &\frac{1}{N_n(H)} \sum_{i=1}^{n} (Y_i - \overline{Y}_H)^2
    \identity_{\bold{X}_i \in H} \\

&- \frac{1}{N_n(H)} \sum_{i=1}^n (Y_i - \overline{Y}_{H_L} \identity_{X_i^{(j)} < \hat{q}_{n,r}^{(j)}}
    - \overline{Y}_{H_R} \identity_{X_i^{(j)} \ge \hat{q}_{n,r}^{(j)}})^2 \identity_{\bold{X}_i \in H},

\end{aligned}
```

where

-  $H$ is some node,
-  $\overline{Y}_H$ is the average of the $Y_i$'s such that $\bold{X}_i \in H$,
-  $N_n(H)$ is the number of data points $\bold{X}_i$ falling into $H$, and
- the empirical $r$-th $q$-quantile $\hat{q}_{n,r}^{(j)}$ of $\{ X_i^{(j)}, ..., X_n^{(j)} \}$ for $r \in \{1, ..., q - 1\}$ is defined in Equation 4.2 by

```math
\newcommand\identity{1\kern-0.25em\text{l}}

\hat{q}_{n,r}^{(j)} = \inf \{ x \in \Reals \: : \: \frac{1}{n} \sum_{i=1}^n \identity_{x_i^{(j)} \le x} \ge \frac{r}{q} \}
```

Note that this definition is for a tree built with the entire dataset without resampling.
When $2 \le q$ ($q$ is typically 10), then the theoretical $q$-quantiles are defined by

```math
q_r^{*(j)} = \inf \{ x \in \Reals \: : \: \mathbb{P} (X^{(j)} \le x) \ge \frac{r}{q} \}.
```

The empircal $r$-th $q$-quantile $\hat{q}_{n,r}^{(j)}$ can be used to determine the splits, namely the left and right subtrees obtained after splitting are defined as

```math
H_L = \{ x \in H \: : \: \bold{x}^{(j)} < \hat{q}_{n,r}^{(j)} \}, \: \: \: H_R = \{ x \in H \: : \: \bold{x}^{(j)} \ge \hat{q}_{n,r}^{(j)} \}.
```

Or in other words, the node splits are restricted to to the $q$-empirical quantiles of the marginals $X^{(1)},...,X^{(p)}$, with typically $q = 10$ (Bénard et al., [2021](http://proceedings.mlr.press/v130/benard21a.html)).
Setting $q = 10$ causes more stable trees because it splits the input space in a grid of $10^p$ hyperrectangles for $p$ features and this number has empirically been found to be accurate.

## Empirical quantiles

The empirical distribution is defined as (Hartmann, [2022](https://www.heinrichhartmann.com/archive/quantiles.html#empirical-quantiles))

```math
P_{emp} = \frac{1}{n} \sum_{i=1}^n \delta_{x_i}.
```

where $x_i$ is contained in the dataset $D = (x_i, ...., x_n)$ of real numbers with $1 \leq n$.

"So a random variable $X_{emp} \sim P_{emp}$ takes values that are uniformly randomly chosen from $D$."

For each empirical $q$-quantile, the following conditions should be met

```math
\frac{\# \{ x \in D \: : \: x < y\}}{n} \leq q \: \: \: \text{and} \: \: \: \frac{\#\{x \in D \: : \: x > y\}}{n} \leq 1 - q.
```
