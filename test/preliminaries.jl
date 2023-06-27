import Base

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

using CategoricalArrays: CategoricalValue, categorical, unwrap
using CSV: CSV
using DataDeps: DataDeps, DataDep, @datadep_str
using Documenter: DocMeta, doctest
using MLDatasets: BostonHousing, Titanic
using DataFrames:
    DataFrames,
    DataFrame,
    Not,
    dropmissing!,
    rename!,
    rename,
    select
using DecisionTree: DecisionTreeClassifier
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
    make_regression,
    rsq,
    predict
using MLJDecisionTreeInterface: DecisionTree
using MLJLinearModels: LinearRegressor
using LightGBM.MLJInterface: LGBMClassifier, LGBMRegressor
using Random: shuffle
using StableRNGs: StableRNG
using SIRUS
using Statistics: mean, var
using Tables: Tables
using Test

const S = SIRUS
_rng(seed::Int=1) = StableRNG(seed)

if !haskey(ENV, "REGISTERED_HABERMAN")
    name = "Haberman"
    message = "Slightly modified copy of Haberman's Survival Data Set"
    remote_path = "https://github.com/rikhuijzer/haberman-survival-dataset/releases/download/v1.0.0/haberman.csv"
    checksum = "a7e9aeb249e11ac17c2b8ea4fdafd5c9392219d27cb819ffaeb8a869eb727a0f"
    DataDeps.register(DataDep(name, message, remote_path, checksum))
    ENV["REGISTERED_HABERMAN"] = "true"
end

"""
Return the Haberman survival dataset.

Accuracy on this dataset can be verified against the benchmark by Nalenz and Augustin (https://proceedings.mlr.press/v151/nalenz22a.html).
In that paper, the AUC for SIRUS and XGBoost are respectively 0.651 and 0.688.
"""
function haberman()
    dir = datadep"Haberman"
    path = joinpath(dir, "haberman.csv")
    df = CSV.read(path, DataFrame)
    df[!, :survival] = categorical(df.survival)
    # Need Floats for the LGBMClassifier.
    for col in [:age, :year, :nodes]
        df[!, col] = float.(df[:, col])
    end
    return df
end

"""
Return the Boston Housing Dataset after changing the outcome to binary.
"""
function boston()
    data = BostonHousing()
    df = hcat(data.features, data.targets)
    dropmissing!(df)
    for col in names(df)
        df[!, col] = float.(df[:, col])
    end
    # Median value of owner-occupied homes in 1000's of dollars.
    target = :MEDV
    m = mean(df[:, target]) # 22.5 thousand dollars.
    # y = categorical([value < m ? 0 : 1 for value in df[:, target]])
    y = df[:, target]
    X = MLJBase.table(MLJBase.matrix(df[:, Not(target)]))
    return (X, y)
end

function _Split(feature::Int, splitval::Float32, direction::Symbol)
    return S.Split(feature, string(feature), splitval, direction)
end

nothing
