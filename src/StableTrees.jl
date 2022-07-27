module StableTrees

import AbstractTrees: children, nodevalue
import Base

using AbstractTrees: AbstractTrees, print_tree
using CategoricalArrays: CategoricalValue, unwrap
using InlineStrings: String255
using LinearAlgebra: rank
using MLJModelInterface: UnivariateFinite
using PrecompileSignatures: @precompile_signatures
using Random: AbstractRNG, default_rng, shuffle
using Statistics: mean, median
using Tables: Tables, matrix

const Float = Float32

export StableForestClassifier, StableRulesClassifier
export feature_names, directions

include("forest.jl")
include("rules.jl")
export TreePath
include("dependent.jl")

include("mlj.jl")
const StableForestClassifier = MLJImplementation.StableForestClassifier
const StableRulesClassifier = MLJImplementation.StableRulesClassifier

@precompile_signatures(StableTrees)

end # module
