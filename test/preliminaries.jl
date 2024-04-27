import Base

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
ENV["CAN_RUN_R_SIRUS"] = v"1.8" < VERSION

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

function _score(e::PerformanceEvaluation)
    return round(only(e.measurement); sigdigits=2)
end

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
Return the Boston Housing Dataset.
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
    y, X = MLJBase.unpack(df, ==(target))
    return (X, y)
end

if !haskey(ENV, "REGISTERED_HEART_DISEASE")
    name = "Heart Disease"
    message = "UCI Heart Disease dataset"
    remote_path = "https://github.com/rikhuijzer/heart-disease-dataset/releases/download/v1.0.0/heart-disease-dataset.csv"
    checksum = "04e11d14886c6470fc4e347ca710136521d7184423238f28161c1d8022ce0c5d"
    DataDeps.register(DataDep(name, message, remote_path, checksum))
    ENV["REGISTERED_HEART_DISEASE"] = "true"
end

"""
Return the Heart Disease dataset.
"""
function heart_disease()
    dir = datadep"Heart Disease"
    path = joinpath(dir, "heart-disease-dataset.csv")
    df = CSV.read(path, DataFrame)
    target = :target
    df[!, target] = categorical(df[:, target])
    y, X = MLJBase.unpack(df, ==(target))
end

function _SubClause(feature::Int, splitval::Float32, direction::Symbol)
    return S.SubClause(feature, string(feature), splitval, direction)
end

datasets = Dict{String,Tuple}(
    "blobs" => let
        n = 200
        p = 40
        make_blobs(n, p; centers=2, rng=_rng(), shuffle=true)
    end,
    "titanic" => let
        titanic = Titanic()
        df = titanic.features
        F = [:Pclass, :Sex, :Age, :SibSp, :Parch, :Fare, :Embarked]
        sub = select(df, F...)
        sub[!, :y] = categorical(titanic.targets[:, 1])
        sub[!, :Sex] = ifelse.(sub.Sex .== "male", 1, 0)
        dropmissing!(sub)
        embarked2int(x) = x == "S" ? 1 : x == "C" ? 2 : 3
        sub[!, :Embarked] = embarked2int.(sub.Embarked)
        X = MLJBase.table(MLJBase.matrix(sub[:, Not(:y)]))
        (X, sub.y)
    end,
    "cancer" => let
        df = cancer()
        X = MLJBase.table(MLJBase.matrix(df[:, Not(:Diagnosis)]))
        (X, df.Diagnosis)
    end,
    "diabetes" => let
        df = diabetes()
        X = MLJBase.table(MLJBase.matrix(df[:, Not(:Outcome)]))
        (X, df.Outcome)
    end,
    "haberman" => let
        df = haberman()
        X = MLJBase.table(MLJBase.matrix(df[:, Not(:survival)]))
        y = categorical(df.survival)
        (X, y)
    end,
    "heart disease" => let
        y, X = heart_disease()
        (X, y)
    end,
    "iris" => let
        iris = Iris()
        X = iris.features
        y = [x == "Iris-setosa" ? 1 : x == "Iris-versicolor" ? 2 : 3 for x in iris.targets.class]
        (X, categorical(y))
     end,
    "boston" => boston(),
    "make_regression" => let
        make_regression(600, 3; noise=0.0, sparse=0.0, outliers=0.0, rng=_rng())
     end
)

results = DataFrame(;
        Dataset=String[],
        Model=String[],
        Hyperparameters=String[],
        measure=String[],
        score=String[],
        se=String[],
        nfolds=Int[]
    )

function _with_trailing_zero(score::Real)::String
    text = string(score)::String
    if length(text) == 3
        return text * '0'
    else
        return text
    end
end

_filter_rng(hyper::NamedTuple) = Base.structdiff(hyper, (; rng=:foo))
_pretty_name(modeltype) = last(split(string(modeltype), '.'))
_hyper2str(hyper::NamedTuple) = hyper == (;) ? "(;)" : string(hyper)::String

function _evaluate(
        model,
        X,
        y;
        nfolds::Number=10,
        measure=auc,
        acceleration=MLJBase.CPUThreads()
    )
    resampling = CV(; nfolds, shuffle=true, rng=_rng())
    evaluate(model, X, y; acceleration, verbosity=0, resampling, measure)
end

function _evaluate!(
        results::DataFrame,
        dataset::String,
        modeltype::DataType,
        hyperparameters::NamedTuple=(; );
        measure=auc,
        acceleration=MLJBase.CPUThreads()
    )
    X, y = datasets[dataset]
    nfolds = 10
    model = modeltype(; hyperparameters...)
    e = _evaluate(model, X, y; nfolds, measure, acceleration)
    score = _with_trailing_zero(_score(e))
    se = let
        val = round(only(MLJBase._standard_errors(e)); digits=2)
        _with_trailing_zero(val)
    end
    measure::String = measure == auc ? "auc" :
        measure == accuracy ? "accuracy" :
        measure == rsq ? "R²" :
        error("Cannot prettify measure $measure")
    row = (;
        Dataset=dataset,
        Model=_pretty_name(modeltype),
        Hyperparameters=_hyper2str(_filter_rng(hyperparameters)),
        measure,
        score,
        se,
        nfolds
    )
    push!(results, row)
    return e
end

nothing
