module SIRUS

import AbstractTrees: children, nodevalue
import Base

using AbstractTrees: AbstractTrees, print_tree
using CategoricalArrays: CategoricalValue, unwrap
using InlineStrings: String255
using LinearAlgebra: rank
using MLJLinearModels: MLJLinearModels, RidgeRegressor, glr
using MLJModelInterface: UnivariateFinite, Probabilistic, fit
using PrecompileSignatures: @precompile_signatures
using Random: AbstractRNG, default_rng, seed!, shuffle
using Statistics: mean, median, std
using Tables: Tables, matrix

export feature_names, directions, satisfies

include("helpers.jl")
using .Helpers: colnames, nfeatures, view_feature

include("empiricalquantiles.jl")
using .EmpiricalQuantiles: Cutpoints, cutpoints
export Cutpoints, cutpoints

include("forest.jl")
include("classification.jl")
include("regression.jl")
include("rules.jl")
include("ruleshow.jl")
include("weights.jl")
export TreePath
include("dependent.jl")

include("mlj.jl")
const StableForestClassifier = MLJImplementation.StableForestClassifier
export StableForestClassifier
const StableRulesClassifier = MLJImplementation.StableRulesClassifier
export StableRulesClassifier
const StableForestRegressor = MLJImplementation.StableForestRegressor
export StableForestRegressor
const StableRulesRegressor = MLJImplementation.StableRulesRegressor
export StableRulesRegressor

@precompile_signatures(SIRUS)

end # module
