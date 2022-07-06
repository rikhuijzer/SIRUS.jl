module StableTrees

import AbstractTrees: children, nodevalue
import Base

using CategoricalArrays: CategoricalValue, unwrap
using LinearAlgebra: norm, rank
using MLJModelInterface: UnivariateFinite
using Random: AbstractRNG, default_rng, shuffle
using Statistics: mean
using Tables: Tables

const Float = Float32

include("forest.jl")
include("rules.jl")
export TreePath
include("dependent.jl")

include("mlj.jl")
const StableForestClassifier = MLJImplementation.StableForestClassifier
export StableForestClassifier
const StableRulesClassifier = MLJImplementation.StableRulesClassifier
export StableRulesClassifier

end # module
