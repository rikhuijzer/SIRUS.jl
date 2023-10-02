import Base

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

using CategoricalArrays:
    CategoricalValue,
    CategoricalVector,
    categorical,
    unwrap
using CSV: CSV
using DataDeps: DataDeps, DataDep, @datadep_str
using Documenter: DocMeta, doctest
using MLDatasets:
    BostonHousing,
    Iris,
    Titanic
using DataFrames:
    DataFrames,
    DataFrame,
    Not,
    dropmissing!,
    rename!,
    rename,
    select
using DecisionTree: DecisionTree
using MLJBase:
    CV,
    MLJBase,
    PerformanceEvaluation,
    evaluate,
    mode,
    fit!,
    machine,
    make_blobs,
    make_moons,
    make_regression,
    predict
using MLJDecisionTreeInterface: DecisionTreeClassifier, DecisionTreeRegressor
using MLJLinearModels: LogisticClassifier, LinearRegressor, MultinomialClassifier
using MLJTestInterface: MLJTestInterface
using MLJXGBoostInterface: XGBoostClassifier, XGBoostRegressor
using Random: shuffle, seed!
using StableRNGs: StableRNG
using StatisticalMeasures:
    accuracy,
    auc,
    rsq
using SIRUS
using Statistics: mean, var
using Tables: Tables
using Test

const S = SIRUS
_rng(seed::Int=1) = StableRNG(seed)

if !haskey(ENV, "REGISTERED_CANCER")
    name = "Cancer"
    message = "Wisconsin Diagnostic Breast Cancer (WDBC) dataset"
    remote_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    checksum = "d606af411f3e5be8a317a5a8b652b425aaf0ff38ca683d5327ffff94c3695f4a"
    DataDeps.register(DataDep(name, message, remote_path, checksum))
    ENV["REGISTERED_CANCER"] = "true"
end

function cancer()
    dir = datadep"Cancer"
    path = joinpath(dir, "wdbc.data")
    df = CSV.read(path, DataFrame; header=false)
    DataFrames.select!(df, Not(1))
    for col in names(df)
        if eltype(df[!, col]) isa AbstractString
            df[!, col] = tryparse.(Float64, df[:, col])
        end
    end
    DataFrames.rename!(df, :Column2 => :Diagnosis)
    df[!, :Diagnosis] = categorical([x == "B" ? 1.0 : 0.0 for x in df.Diagnosis])
    return df
end

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

if !haskey(ENV, "REGISTERED_DIABETES")
    name = "Diabetes"
    message = "Pima Indians Diabetes Database"
    remote_path = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    checksum = "6bfe5d0f379d17a0e0819b996407e3c09bf80febd4287f2ed212190dfff154af"
    DataDeps.register(DataDep(name, message, remote_path, checksum))
    ENV["REGISTERED_DIABETES"] = "true"
end

function diabetes()
    dir = datadep"Diabetes"
    path = joinpath(dir, "pima-indians-diabetes.data.csv")
    header = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
        "Outcome",
    ]
    df = CSV.read(path, DataFrame; header)
    for col in names(df)
        df[!, col] = float.(df[:, col])
    end
    df[!, :Outcome] = categorical(df.Outcome)
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
