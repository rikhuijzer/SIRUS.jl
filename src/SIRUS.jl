module SIRUS

import AbstractTrees: children, nodevalue
import Base

using AbstractTrees: AbstractTrees, print_tree
using CategoricalArrays: CategoricalValue, unwrap
using InlineStrings: String255
using LinearAlgebra: rank
using MLJBase: mode
using MLJLinearModels: MLJLinearModels, RidgeRegressor, glr
using MLJModelInterface: Deterministic, UnivariateFinite, Probabilistic, fit
using Random: AbstractRNG, default_rng, seed!, shuffle
using OrderedCollections: OrderedSet
using Statistics: mean, median
using Tables: Tables, matrix


include("helpers.jl")
using .Helpers: colnames, nfeatures, view_feature

include("empiricalquantiles.jl")
using .EmpiricalQuantiles: Cutpoints, cutpoints
export Cutpoints, cutpoints

include("forest.jl")
export StableForest
include("classification.jl")
include("regression.jl")
include("rules.jl")
export StableRules, feature_names, directions, satisfies
include("ruleshow.jl")
include("dependent.jl")
include("weights.jl")
include("extract.jl")
export feature_importance, feature_importances

include("mlj.jl")
const StableForestClassifier = MLJImplementation.StableForestClassifier
export StableForestClassifier
const StableRulesClassifier = MLJImplementation.StableRulesClassifier
export StableRulesClassifier
const StableForestRegressor = MLJImplementation.StableForestRegressor
export StableForestRegressor
const StableRulesRegressor = MLJImplementation.StableRulesRegressor
export StableRulesRegressor

end # module
