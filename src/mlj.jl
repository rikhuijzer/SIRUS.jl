module MLJImplementation

import MLJModelInterface:
    fit,
    matrix,
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
using StableTrees:
    DEFAULT_PENALTY,
    StableForest,
    StableRules,
    _forest,
    _mean,
    _predict,
    _rules,
    _process_rules
using Tables: Tables

"""
    StableForestClassifier <: MLJModelInterface.Probabilistic

Random forest classifier with a stabilized forest structure (Bénard et al., [2021](http://proceedings.mlr.press/v130/benard21a.html)).
This stabilization increases stability when extracting rules.
The impact on the predictive accuracy compared to standard random forests should be relatively small.
"""
Base.@kwdef mutable struct StableForestClassifier <: Probabilistic
    rng::AbstractRNG=default_rng()
    partial_sampling::Real=0.7
    n_trees::Int=1_000
    max_depth::Int=2
    q::Int=10
    min_data_in_leaf::Int=5
end

Base.@kwdef mutable struct StableRulesClassifier <: Probabilistic
    rng::AbstractRNG=default_rng()
    partial_sampling::Real=0.7
    n_trees::Int=1_000
    max_depth::Int=2
    q::Int=10
    min_data_in_leaf::Int=5
    max_rules::Int=10
    weight_penalty::Float64=DEFAULT_PENALTY
end

metadata_model(
    StableForestClassifier;
    input_scitype=Table(Continuous, Count),
    target_scitype=AbstractVector{<:Finite},
    supports_weights=false,
    docstring="Random forest classifier with a stabilized forest structure",
    path="StableTrees.StableForestClassifier"
)

metadata_model(
    StableRulesClassifier;
    input_scitype=Table(Continuous, Count),
    target_scitype=AbstractVector{<:Finite},
    supports_weights=false,
    docstring="Stable rule-based classifier",
    path="StableTrees.StableForestClassifier"
)

metadata_pkg.(
    [StableForestClassifier, StableRulesClassifier];
    name="StableTrees",
    uuid="9113e207-2504-4b06-8eee-d78e288bee65",
    url="https://github.com/rikhuijzer/StableTrees.jl",
    julia=true,
    license="MIT",
    is_wrapper=false
)

function _float(T::Type{<:AbstractString}, A)
    @warn """
        Converting $(typeof(A)) to floats via `levelcode.(A)`.

        Consider passing a (Categorical)Vector of `Float`s to avoid mixing up classes.
        """
    return categorical(levelcode.(A))
end
_float(T, A) = convert(AbstractArray{typeof(float(zero(T)))}, A)

"""
Return a floating point vector of `A`.
This method patches the version from CategoricalArrays.jl for `AbstractString`s.
"""
function _float(A::CategoricalArray{T}) where T
    if !isconcretetype(T)
        error("`float` not defined on abstractly-typed arrays; please convert to a more specific type")
    end
    return _float(T, A)
end

function fit(model::StableForestClassifier, verbosity::Int, X, y)
    forest = _forest(
        model.rng,
        matrix(X),
        _float(y);
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

function predict(model::StableForestClassifier, fitresult::StableForest, Xnew)
    forest = fitresult
    return _predict(forest, matrix(Xnew))
end

function fit(model::StableRulesClassifier, verbosity::Int, X, y)
    forest = _forest(
        model.rng,
        matrix(X),
        _float(y);
        model.partial_sampling,
        model.n_trees,
        model.max_depth,
        model.q,
        model.min_data_in_leaf
    )
    penalty = model.weight_penalty
    fitresult = StableRules(forest, model.max_rules; penalty)
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

function predict(model::StableRulesClassifier, fitresult::StableRules, Xnew)
    isempty(fitresult.rules) && error("Zero rules")
    return _predict(fitresult, matrix(Xnew))
end

end # module
