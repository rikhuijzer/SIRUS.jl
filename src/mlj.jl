module MLJImplementation

import MLJModelInterface: fit, metadata_model, metadata_pkg

using MLJModelInterface: MLJModelInterface, Continuous, Finite, Probabilistic, Table
using Random: AbstractRNG, default_rng
using StableTrees: _forest

"""
    StableForestClassifier <: MLJModelInterface.Probabilistic

Random forest classifier with a stabilized forest structure (Bénard et al., [2021](http://proceedings.mlr.press/v130/benard21a.html)).
This stabilization increases stability when extracting rules with only a small impact on the predictive accuracy compared to standard random forests.
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

end # module
