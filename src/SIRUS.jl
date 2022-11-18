module SIRUS

import AbstractTrees: children, nodevalue
import Base

using AbstractTrees: AbstractTrees, print_tree
using CategoricalArrays: CategoricalValue, unwrap
using InlineStrings: String255
using LinearAlgebra: rank
using MLJLinearModels: MLJLinearModels, ElasticNetRegressor, glr
using MLJModelInterface: UnivariateFinite, Probabilistic, fit
using PrecompileSignatures: @precompile_signatures
using Random: AbstractRNG, default_rng, seed!, shuffle
using Statistics: mean, median
using Tables: Tables, matrix

const Float = Float32

export StableForestClassifier, StableRulesClassifier
export feature_names, directions, satisfies

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
