module StableTrees

import AbstractTrees: children, nodevalue

using Random: AbstractRNG

const Float = Float32

export StableForestClassifier

include("forest.jl")
include("interface.jl")

end # module
