module MLJImplementation

import MLJModelInterface:
    fit,
    predict,
    metadata_model,
    metadata_pkg

using CategoricalArrays:
    CategoricalArray,
    CategoricalValue,
    categorical,
    levelcode,
    unwrap
using MLJModelInterface:
    MLJModelInterface,
    UnivariateFinite,
    Continuous,
    Count,
    Finite,
    Probabilistic,
    Table
using Random: AbstractRNG, default_rng
using SIRUS:
    Algorithm,
    Classification,
    Regression,
    StableForest,
    StableRules,
    colnames,
    _forest,
    _mean,
    _predict,
    _rules,
    _process_rules
using Tables: Tables, matrix

"""
    StableForestClassifier(;
        rng::AbstractRNG=default_rng(),
        partial_sampling::Real=0.7,
        n_trees::Int=1_000,
        max_depth::Int=2,
        q::Int=10,
        min_data_in_leaf::Int=5
    ) <: MLJModelInterface.Probabilistic

Random forest classifier with a stabilized forest structure (Bénard et al., [2021](http://proceedings.mlr.press/v130/benard21a.html)).
This stabilization increases stability when extracting rules.
The impact on the predictive accuracy compared to standard random forests should be relatively small.

!!! note
    Just like normal random forests, this model is not easily explainable.
    If you are interested in an explainable model, use the `StableRulesClassifier`.

# Example

The classifier satisfies the MLJ interface, so it can be used like any other MLJ model.
For example, it can be used to create a machine:

```julia
julia> using SIRUS, MLJ

julia> mach = machine(StableForestClassifier(), X, y);
```

# Arguments

- `rng`: Random number generator. `StableRNGs` are advised.
- `partial_sampling`:
    Ratio of samples to use in each subset of the data.
    The default of 0.7 should be fine for most cases.
- `n_trees`:
    The number of trees to use.
    It is advisable to use at least thousand trees to for a better rule selection, and
    in turn better predictive performance.
- `max_depth`:
    The depth of the tree.
    A lower depth decreases model complexity and can therefore improve accuracy when the sample size is small (reduce overfitting).
- `q`: Number of cutpoints to use per feature.
    The default value of 10 should be good for most situations.
- `min_data_in_leaf`: Minimum number of data points per leaf.
"""
Base.@kwdef mutable struct StableForestClassifier <: Probabilistic
    rng::AbstractRNG=default_rng()
    partial_sampling::Real=0.7
    n_trees::Int=1_000
    max_depth::Int=2
    q::Int=10
    min_data_in_leaf::Int=5
end

"""
    StableRulesClassifier(;
        rng::AbstractRNG=default_rng(),
        partial_sampling::Real=0.7,
        n_trees::Int=1_000,
        max_depth::Int=2,
        q::Int=10,
        min_data_in_leaf::Int=5,
        max_rules::Int=10
    ) -> MLJModelInterface.Probabilistic

Explainable rule-based model based on a random forest.
This SIRUS algorithm extracts rules from a stabilized random forest.
See the [main page of the documentation](https://huijzer.xyz/StableTrees.jl/dev/) for details about how it works.

# Example

The classifier satisfies the MLJ interface, so it can be used like any other MLJ model.
For example, it can be used to create a machine:

```julia
julia> using SIRUS, MLJ

julia> mach = machine(StableRulesClassifier(; max_rules=15), X, y);
```

# Arguments

- `rng`: Random number generator. `StableRNGs` are advised.
- `partial_sampling`:
    Ratio of samples to use in each subset of the data.
    The default of 0.7 should be fine for most cases.
- `n_trees`:
    The number of trees to use.
    The higher the number, the more likely it is that the correct rules are extracted from the trees, but also the longer model fitting will take.
    In most cases, 1000 rules should be more than enough, but it might be useful to run 2000 rules one time and verify that the model performance does not change much.
- `max_depth`:
    The depth of the tree.
    A lower depth decreases model complexity and can therefore improve accuracy when the sample size is small (reduce overfitting).
- `q`: Number of cutpoints to use per feature.
    The default value of 10 should be good for most situations.
- `min_data_in_leaf`: Minimum number of data points per leaf.
- `max_rules`:
    This is the most important hyperparameter.
    In general, the more rules, the more accurate the model.
    However, more rules will also decrease model interpretability.
    So, it is important to find a good balance here.
    In most cases, 10-40 rules should provide reasonable accuracy while remaining interpretable.
- `lambda`:
    The weights of the final rules are determined via a regularized regression over each rule as a binary feature.
    This hyperparameter specifies the strength of the ridge (L2) regularizer.
    Since the rules are quite strongly correlated, the ridge regularizer is the most useful to stabilize the weight estimates.
"""
Base.@kwdef mutable struct StableRulesClassifier <: Probabilistic
    rng::AbstractRNG=default_rng()
    partial_sampling::Real=0.7
    n_trees::Int=1_000
    max_depth::Int=2
    q::Int=10
    min_data_in_leaf::Int=5
    max_rules::Int=10
    lambda::Float64=5
end

"""
    StableForestRegressor(;
        rng::AbstractRNG=default_rng(),
        partial_sampling::Real=0.7,
        n_trees::Int=1_000,
        max_depth::Int=2,
        q::Int=10,
        min_data_in_leaf::Int=5
    ) <: MLJModelInterface.Probabilistic

Random forest regressor with a stabilized forest structure (Bénard et al., [2021](http://proceedings.mlr.press/v130/benard21a.html)).
See the documentation for the `StableForestClassifier` for more information about the hyperparameters.

# Example

The classifier satisfies the MLJ interface, so it can be used like any other MLJ model.
For example, it can be used to create a machine:

```julia
julia> using SIRUS, MLJ

julia> mach = machine(StableForestRegressor(), X, y);
```
"""
Base.@kwdef mutable struct StableForestRegressor <: Probabilistic
    rng::AbstractRNG=default_rng()
    partial_sampling::Real=0.7
    n_trees::Int=1_000
    max_depth::Int=2
    q::Int=10
    min_data_in_leaf::Int=5
end

"""
    StableRulesRegressor(;
        rng::AbstractRNG=default_rng(),
        partial_sampling::Real=0.7,
        n_trees::Int=1_000,
        max_depth::Int=2,
        q::Int=10,
        min_data_in_leaf::Int=5,
        max_rules::Int=10
    ) -> MLJModelInterface.Probabilistic

Explainable rule-based regression model based on a random forest.
See the documentation for the `StableRulesClassifier` for more information about
the model and the hyperparameters.
"""
Base.@kwdef mutable struct StableRulesRegressor <: Probabilistic
    rng::AbstractRNG=default_rng()
    partial_sampling::Real=0.7
    n_trees::Int=1_000
    max_depth::Int=2
    q::Int=10
    min_data_in_leaf::Int=5
    max_rules::Int=10
    lambda::Float64=5
end

metadata_model(
    StableForestClassifier;
    input_scitype=Table(Continuous, Count),
    target_scitype=AbstractVector{<:Finite},
    supports_weights=false,
    docstring="Random forest classifier with a stabilized forest structure",
    path="SIRUS.StableForestClassifier"
)

metadata_model(
    StableRulesClassifier;
    input_scitype=Table(Continuous, Count),
    target_scitype=AbstractVector{<:Finite},
    supports_weights=false,
    docstring="Stable and Interpretable RUle Sets (SIRUS) classifier",
    path="SIRUS.StableForestClassifier"
)

metadata_model(
    StableForestRegressor;
    input_scitype=Table(Continuous, Count),
    target_scitype=AbstractVector{<:Continuous},
    supports_weights=false,
    docstring="Random forest regressor with a stabilized forest structure",
    path="SIRUS.StableForestRegressor"
)

metadata_model(
    StableRulesRegressor;
    input_scitype=Table(Continuous, Count),
    target_scitype=AbstractVector{<:Continuous},
    supports_weights=false,
    docstring="Stable and Interpretable RUle Sets (SIRUS) regressor",
    path="SIRUS.StableForestRegressor"
)

metadata_pkg.(
    [
        StableForestClassifier,
        StableRulesClassifier,
        StableForestRegressor,
        StableRulesRegressor
    ];
    name="SIRUS",
    uuid="9113e207-2504-4b06-8eee-d78e288bee65",
    url="https://github.com/rikhuijzer/SIRUS.jl",
    julia=true,
    license="MIT",
    is_wrapper=false
)

"""
Return a floating point vector of `A`.
This method patches the version from CategoricalArrays.jl for `AbstractString`s.
"""
function _float(A::CategoricalArray{T}) where T
    if !isconcretetype(T)
        msg = "`float` not defined on abstractly-typed arrays; please convert to a more specific type"
        throw(ArgumentError(msg))
    end
    if T isa Type{String}
        msg = "Cannot automatically convert $(typeof(A)) to an array containing `Float`s."
        throw(ArgumentError(msg))
    end
    return float(A)
end
_float(A::AbstractVector) = float.(A)

function fit(
        model::Union{StableForestClassifier, StableForestRegressor},
        algo::Algorithm,
        verbosity::Int,
        X,
        y
    )
    forest = _forest(
        model.rng,
        algo,
        matrix(X),
        _float(y),
        colnames(X);
        model.partial_sampling,
        model.n_trees,
        model.max_depth,
        model.q,
        model.min_data_in_leaf
    )
    fitresult = forest
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

function fit(model::StableForestClassifier, verbosity::Int, X, y)
    algo = Classification()
    return fit(model, algo, verbosity, X, y)
end

function fit(model::StableForestRegressor, verbosity::Int, X, y)
    algo = Regression()
    return fit(model, algo, verbosity, X, y)
end

function predict(
        model::Union{StableForestClassifier, StableForestRegressor},
        fitresult::StableForest,
        Xnew
    )
    forest = fitresult
    return _predict(forest, matrix(Xnew))
end

function fit(
        model::Union{StableRulesClassifier, StableRulesRegressor},
        algo::Algorithm,
        verbosity::Int,
        X,
        y
    )
    data = matrix(X)
    outcome = _float(y)
    forest = _forest(
        model.rng,
        algo,
        data,
        outcome,
        colnames(X);
        model.partial_sampling,
        model.n_trees,
        model.max_depth,
        model.q,
        model.min_data_in_leaf
    )
    fitresult = StableRules(forest, data, outcome, model)
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

function fit(model::StableRulesClassifier, verbosity::Int, X, y)
    algo = Classification()
    fit(model, algo, verbosity, X, y)
end

function fit(model::StableRulesRegressor, verbosity::Int, X, y)
    algo = Regression()
    fit(model, algo, verbosity, X, y)
end

function predict(
        model::StableRulesClassifier,
        fitresult::StableRules,
        Xnew
    )
    isempty(fitresult.rules) && error("Zero rules")
    return _predict(fitresult, matrix(Xnew))
end

function predict(
        model::StableRulesRegressor,
        fitresult::StableRules,
        Xnew
    )
    isempty(fitresult.rules) && error("Zero rules")
    predictions = _predict(fitresult, matrix(Xnew))
    unpacked = [only(p) for p in predictions]
    return unpacked
end

end # module
