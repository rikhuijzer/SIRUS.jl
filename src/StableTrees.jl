module StableTrees

import AbstractTrees: children, nodevalue

using CategoricalArrays: CategoricalValue, unwrap
using Random: AbstractRNG
using Tables: Tables

const Float = Float32

include("forest.jl")

include("mlj.jl")
const StableForestClassifier = MLJImplementation.StableForestClassifier
export StableForestClassifier

end # module
