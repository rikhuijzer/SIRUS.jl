import Base

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

using CategoricalArrays: CategoricalValue, categorical, unwrap
using MLDatasets: Titanic
using DataFrames: DataFrames, Not, select, dropmissing!
using DecisionTree: DecisionTree
using MLJBase:
    CV,
    MLJBase,
    PerformanceEvaluation,
    accuracy,
    auc,
    evaluate,
    mode,
    fit!,
    machine,
    make_blobs,
    make_moons,
    predict
using LightGBM.MLJInterface: LGBMClassifier
using StableRNGs: StableRNG
using StableTrees
using Statistics: mean
using Tables: Tables
using Test

ST = StableTrees
Float = ST.Float
_rng(; seed::Int=1) = StableRNG(seed)

