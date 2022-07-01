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
using StableTrees: _forest, _predict
using Statistics: mean
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

metadata_model(
    StableForestClassifier;
    input_scitype=Table(Continuous),
    target_scitype=AbstractVector{<:Finite},
    supports_weights=false,
    docstring="Random forest classifier with a stabilized forest structure",
    path="StableTrees.StableForestClassifier"
)

metadata_pkg.(
    [StableForestClassifier];
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

function _mode(y::AbstractVector)
    @assert !isempty(y)
    # The number of occurences for each unique element in y.
    counts = Dict{Any,Int}()
    # The first index of each unique element in y.
    # This ensures that the return type is the same as input type.
    indexes = Dict{Any,Int}()
    for (i, e) in enumerate(y)
        if e in keys(counts)
            counts[e] += 1
        else
            counts[e] = 0
            indexes[e] = i
        end
    end
    max_counted_index = 1
    max_count = 0
    for e in keys(counts)
        count = counts[e]
        if max_count < count
            max_counted_index = indexes[e]
            max_count = count
        end
    end
    return y[max_counted_index]
end

function predict(model::StableForestClassifier, fitresult, Xnew)
    forest = fitresult
    probs = map(Tables.rows(Xnew)) do row
        probs = [_predict(tree, row) for tree in forest.trees]
        only(mean(probs; dims=1))
    end
    P = reduce(hcat, probs)'
    return UnivariateFinite(forest.classes, P; pool=missing)
end

end # module
