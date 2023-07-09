module MLJImplementation

import MLJModelInterface:
    MLJModelInterface,
    fit,
    predict,
    metadata_model,
    metadata_pkg

const MMI = MLJModelInterface

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

const PARTIAL_SAMPLING_DEFAULT = 0.7
const N_TREES_DEFAULT = 1000
const MAX_DEPTH_DEFAULT = 2
const Q_DEFAULT = 10
const MIN_DATA_IN_LEAF_DEFAULT = 5
const MAX_RULES_DEFAULT = 10
const LAMBDA_DEFAULT = 1.0

MMI.@mlj_model mutable struct StableForestClassifier <: Probabilistic
    rng::AbstractRNG=default_rng()
    partial_sampling::Real=PARTIAL_SAMPLING_DEFAULT
    n_trees::Int=N_TREES_DEFAULT
    max_depth::Int=MAX_DEPTH_DEFAULT
    q::Int=Q_DEFAULT
    min_data_in_leaf::Int=MIN_DATA_IN_LEAF_DEFAULT
end

MMI.@mlj_model mutable struct StableRulesClassifier <: Probabilistic
    rng::AbstractRNG=default_rng()
    partial_sampling::Real=PARTIAL_SAMPLING_DEFAULT
    n_trees::Int=N_TREES_DEFAULT
    max_depth::Int=MAX_DEPTH_DEFAULT
    q::Int=Q_DEFAULT
    min_data_in_leaf::Int=MIN_DATA_IN_LEAF_DEFAULT
    max_rules::Int=MAX_RULES_DEFAULT
    lambda::Float64=LAMBDA_DEFAULT
end

MMI.@mlj_model mutable struct StableForestRegressor <: Probabilistic
    rng::AbstractRNG=default_rng()
    partial_sampling::Real=PARTIAL_SAMPLING_DEFAULT
    n_trees::Int=N_TREES_DEFAULT
    max_depth::Int=MAX_DEPTH_DEFAULT
    q::Int=Q_DEFAULT
    min_data_in_leaf::Int=MIN_DATA_IN_LEAF_DEFAULT
end

MMI.@mlj_model mutable struct StableRulesRegressor <: Probabilistic
    rng::AbstractRNG=default_rng()
    partial_sampling::Real=PARTIAL_SAMPLING_DEFAULT
    n_trees::Int=N_TREES_DEFAULT
    max_depth::Int=MAX_DEPTH_DEFAULT
    q::Int=Q_DEFAULT
    min_data_in_leaf::Int=MIN_DATA_IN_LEAF_DEFAULT
    max_rules::Int=MAX_RULES_DEFAULT
    lambda::Float64=LAMBDA_DEFAULT
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
        uniques = sort(unique(A))
        A = collect(0.0:float(length(uniques) - 1))
        # Workaround for https://github.com/rikhuijzer/SIRUS.jl/issues/24.
        @info "Converting outcome classes $(uniques) to $(A)."
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
        model::Union{StableRulesClassifier, StableRulesRegressor},
        fitresult::StableRules,
        Xnew
    )
    isempty(fitresult.rules) && error("Zero rules")
    return _predict(fitresult, matrix(Xnew))
end

const TRAINING_DATA_SECTION = """
# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X, y)

where

- `X`: any table of input features (eg, a `DataFrame`) whose columns
  each have one of the following element scitypes: `Continuous`,
  `Count`, or `<:OrderedFactor`; check column scitypes with `schema(X)`

- `y`: the target, which can be any `AbstractVector` whose element
  scitype is `<:OrderedFactor` or `<:Multiclass`; check the scitype
  with `scitype(y)`

Train the machine with `fit!(mach, rows=...)`.
"""

const HYPERPARAMETERS_SECTION = """
# Hyperparameters

- `rng::AbstractRNG=default_rng()`: Random number generator.
    Using a `StableRNG` from `StableRNGs.jl` is advised.
- `partial_sampling::Float64=$PARTIAL_SAMPLING_DEFAULT`:
    Ratio of samples to use in each subset of the data.
    The default should be fine for most cases.
- `n_trees::Int=$N_TREES_DEFAULT`:
    The number of trees to use.
    It is advisable to use at least thousand trees to for a better rule selection, and
    in turn better predictive performance.
- `max_depth::Int=$MAX_DEPTH_DEFAULT`:
    The depth of the tree.
    A lower depth decreases model complexity and can therefore improve accuracy when the sample size is small (reduce overfitting).
- `q::Int=$Q_DEFAULT`: Number of cutpoints to use per feature.
    The default value should be fine for most situations.
- `min_data_in_leaf::Int=$MIN_DATA_IN_LEAF_DEFAULT`: Minimum number of data points per leaf.
"""

const RULES_HYPERPARAMETERS_SECTION = """
- `max_rules::Int=$MAX_RULES_DEFAULT`:
    This is the most important hyperparameter.
    In general, the more rules, the more accurate the model.
    However, more rules will also decrease model interpretability.
    So, it is important to find a good balance here.
    In most cases, 10 to 40 rules should provide reasonable accuracy while remaining interpretable.
- `lambda::Float64=$LAMBDA_DEFAULT`:
    The weights of the final rules are determined via a regularized regression over each rule as a binary feature.
    This hyperparameter specifies the strength of the ridge (L2) regularizer.
"""

"""
$(MMI.doc_header(StableForestClassifier))

`StableForestClassifier` implements the random forest classifier with a stabilized forest structure (Bénard et al., [2021](http://proceedings.mlr.press/v130/benard21a.html)).
This stabilization increases stability when extracting rules.
The impact on the predictive accuracy compared to standard random forests should be relatively small.

!!! note
    Just like normal random forests, this model is not easily explainable.
    If you are interested in an explainable model, use the `StableRulesClassifier` or
    `StableRulesRegressor`.

$TRAINING_DATA_SECTION

$HYPERPARAMETERS_SECTION
"""
StableForestClassifier

"""
$(MMI.doc_header(StableRulesClassifier))

`StableRulesClassifier` implements the explainable rule-based model based on a random forest.

$TRAINING_DATA_SECTION

$HYPERPARAMETERS_SECTION
$RULES_HYPERPARAMETERS_SECTION
"""
StableRulesClassifier

"""
$(MMI.doc_header(StableForestRegressor))

`StableForestRegressor` implements the random forest regressor with a stabilized forest structure (Bénard et al., [2021](http://proceedings.mlr.press/v130/benard21a.html)).

$TRAINING_DATA_SECTION

$HYPERPARAMETERS_SECTION
"""
StableForestRegressor

"""
$(MMI.doc_header(StableRulesRegressor))

`StableRulesRegressor` implements the explainable rule-based regression model based on a random forest.

$TRAINING_DATA_SECTION

$HYPERPARAMETERS_SECTION
$RULES_HYPERPARAMETERS_SECTION
"""
StableRulesRegressor

end # module
