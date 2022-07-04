module MLJImplementation

import MLJModelInterface:
    fit,
    predict,
    metadata_model,
    metadata_pkg

using CategoricalArrays: CategoricalValue, categorical, unwrap
using MLJModelInterface:
    MLJModelInterface,
    UnivariateFinite,
    Continuous,
    Finite,
    Probabilistic,
    Table
using Random: AbstractRNG, default_rng
using StableTrees:
    _forest,
    _mean_probabilities,
    _predict,
    _rules,
    _select_rules,
    _treat_rules
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
end

metadata_model(
    StableForestClassifier;
    input_scitype=Table(Continuous),
    target_scitype=AbstractVector{<:Finite},
    supports_weights=false,
    docstring="Random forest classifier with a stabilized forest structure",
    path="StableTrees.StableForestClassifier"
)

metadata_model(
    StableRulesClassifier;
    input_scitype=Table(Continuous),
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

function fit(model::StableForestClassifier, verbosity::Int, X, y)
    forest = _forest(
        model.rng,
        X,
        y;
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

function predict(model::StableForestClassifier, fitresult, Xnew)
    forest = fitresult
    return _predict(forest, Xnew)
end

function fit(model::StableRulesClassifier, verbosity::Int, X, y)
    forest = _forest(
        model.rng,
        X,
        y;
        model.partial_sampling,
        model.n_trees,
        model.max_depth,
        model.q,
        model.min_data_in_leaf
    )
    rules = _rules(forest)
    selected = _select_rules(rules)
    U = unique(selected)
    treated = _treat_rules(U)
    cache = nothing
    report = nothing
    return (treated, forest.classes), cache, report
end

function predict(model::StableRulesClassifier, fitresult, Xnew)
    rules, classes = fitresult
    probs = map(Tables.rows(Xnew)) do row
        probs = _predict(rules, row)
    end
    P = reduce(hcat, probs)'
    return UnivariateFinite(classes, P; pool=missing)
end

end # module
