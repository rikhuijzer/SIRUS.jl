![Visual representation of the algorithm which converts decision trees to rule sets. Created with DALL·E 3 and Photopea](https://sirus.jl.huijzer.xyz/dev/image/sirus-with-text.webp)

<h1 align="center">SIRUS.jl</h1>

<p align="center">
    <a href="https://github.com/rikhuijzer/SIRUS.jl/actions?query=workflow%3ACI+branch%3Amain">
        <img src="https://github.com/rikhuijzer/SIRUS.jl/workflows/CI/badge.svg" alt="CI">
    </a>
    <a href="https://github.com/invenia/BlueStyle">
        <img src="https://img.shields.io/badge/Code%20Style-Blue-4495d1.svg" alt="Code Style Blue">
    </a>
    <a style="border-width:0" href="https://doi.org/10.21105/joss.05786">
        <img src="https://joss.theoj.org/papers/10.21105/joss.05786/status.svg" alt="DOI badge" >
    </a>
</p>

<br>

This package is a pure Julia implementation of the **S**table and **I**nterpretable **RU**le **S**ets (SIRUS) algorithm.
The algorithm was originally created by Clément Bénard, Gérard Biau, Sébastien Da Veiga, and Erwan Scornet (Bénard et al., [2021](http://proceedings.mlr.press/v130/benard21a.html)).
`SIRUS.jl` has implemented both classification and regression, but we found that performance is generally best on classification tasks.

The main benefit of this algorithm is that it is **fully explainable**.
This differs from model-agnostic explainability techniques such as SHAP, which convert the model to a simplified representation.
However, **the complex model is still used for predictions**, which can lead to hidden biases or reliability issues.
The SIRUS algorithm fixes this by using a simplified model for **both** for prediction and explanation.

# Installation

```julia
julia> ]

pkg> add SIRUS
```

# Getting Started

This package defines two rule-based models that satisfy the Machine Learning Julia [`MLJ.jl`](https://github.com/alan-turing-institute/MLJ.jl) interface.
The models are `StableRulesClassifier` and `StableRulesRegressor`:

## Example

```julia
julia> using MLJ, SIRUS

julia> X, y = make_blobs(200, 10; centers=2);

julia> X
Tables.MatrixTable{Matrix{Float64}} with 200 rows, 10 columns, and schema:
 :x1   Float64
 :x2   Float64
 :x3   Float64
 :x4   Float64
 :x5   Float64
 :x6   Float64
 :x7   Float64
 :x8   Float64
 :x9   Float64
 :x10  Float64

julia> y
200-element CategoricalArrays.CategoricalArray{Int64,1,UInt32}:
 2
 1
 1
 ⋮
 2
 1
 2

julia> model = StableRulesClassifier();

julia> mach = machine(model, X, y);

julia> fit!(mach);

julia> mach.fitresult
StableRules model with 7 rules:
 if X[i, :x5] < -1.552594 then 0.129 else 0.0 +
 if X[i, :x8] < 0.72402614 then 0.117 else 0.0 +
 if X[i, :x2] < 7.1123967 then 0.123 else 0.0 +
 if X[i, :x8] < 8.840833 then 0.115 else 0.0 +
 if X[i, :x9] < 7.985747 then 0.0 else 0.001 +
 if X[i, :x7] < 6.4651833 then 0.107 else 0.0 +
 if X[i, :x7] < 2.2220817 then 0.119 else 0.024
and 2 classes: [1, 2].
Note: showing only the probability for class 2 since class 1 has probability 1 - p.
```

See `?StableRulesClassifier`, `?StableRulesRegressor`, or the [API documentation](https://sirus.jl.huijzer.xyz/dev/api/) for more information about the models and their hyperparameters.
A full guide through binary classification can be found in the [Simple Binary Classification](https://sirus.jl.huijzer.xyz/dev/binary-classification/) example.

# Documentation

Documentation is at [sirus.jl.huijzer.xyz](https://sirus.jl.huijzer.xyz).

# Contributing

Thank you for your interest in contributing to SIRUS.jl!
There are multiple ways to contribute.

## Questions and Bug Reports

For questions or bug reports, you can open an [issue](https://github.com/rikhuijzer/SIRUS.jl/issues).
Questions can also be asked at the [Julia forum](https://discourse.julialang.org/) or by sending a mail to [github@huijzer.xyz](mailto:github@huijzer.xyz).
Tag `@rikh` in the forum to ensure a quick reply.

## Pull Requests

To submit patches, use pull requests (PRs) here on GitHub.
In general:

- Try to keep PRs limited to one feature or bug; otherwise they become hard to review/verify.
- Try to use the code style that is used in the rest of the codebase.
  See also the [Code Style Blue](https://github.com/invenia/BlueStyle).
- Try to update documentation when updating code, but feel free to leave documentation updates for a separate PR.
- When possible, make PRs as easily reversible as possible.
  Any change that would be easily reversible later provides little risk and can, therefore, more easily be merged.

As long as the PR moves the codebase forward, merging will likely happen.
