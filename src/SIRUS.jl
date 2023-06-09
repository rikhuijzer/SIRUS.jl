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
using Statistics: mean, median
using Tables: Tables, matrix

export StableForestClassifier, StableRulesClassifier
export feature_names, directions, satisfies

include("helpers.jl")
using .Helpers: nfeatures, view_feature

include("empiricalquantiles.jl")
using .EmpiricalQuantiles: Cutpoints, cutpoints
export Cutpoints, cutpoints

include("forest.jl")
include("rules.jl")
include("weights.jl")
export TreePath
include("dependent.jl")

include("mlj.jl")
const StableForestClassifier = MLJImplementation.StableForestClassifier
const StableRulesClassifier = MLJImplementation.StableRulesClassifier

@precompile_signatures(SIRUS)

end # module
