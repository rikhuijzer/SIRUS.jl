#
# This file defines the MLJ wrappers around R sirus and tests them.
# Actual comparisons against other models are done in test/mlj.jl.
#

import MLJModelInterface:
    MLJModelInterface,
    fit,
    predict,
    metadata_model,
    metadata_pkg

using CategoricalArrays:
    CategoricalArray,
    CategoricalPool,
    CategoricalValue
using MLJModelInterface:
    MLJModelInterface,
    UnivariateFinite,
    Continuous,
    Count,
    Deterministic,
    Finite,
    Probabilistic,
    Table
using RCall

const MMI = MLJModelInterface

# @rlibrary sirus

MMI.@mlj_model mutable struct RSirusRegressor <: Deterministic
    max_depth::Int=2
    max_rules::Int=10
end

R"library('sirus')"

n = 100
A = rand(_rng(), n)
B = rand(_rng(), n)
X = DataFrame(; A, B)
y = rand(_rng(), n)

function fit(
        model::RSirusRegressor,
        verbosity::Int,
        X,
        y
    )
    if !Tables.istable(X)
        error("Expected a Table but got $(typeof(Xnew))")
    end
    df = DataFrame(X)
    fitted_model = R"""
        fitted.model <- sirus.fit(
            $df,
            $y,
            type="reg",
            num.rule=$(model.max_rules),
            p0=NULL,
            num.rule.max=$(model.max_rules),
            q=4,
            max.depth=$(model.max_depth),
            num.trees=NULL,
            num.threads=1,
            verbose=FALSE,
            seed=1
        )
        # print(sirus.print(fitted.model))
        fitted.model
    """
    fitresult = fitted_model
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

verbosity = 0

model = RSirusRegressor()
mach = machine(model, X, y, verbosity)
fit!(mach; verbosity)
# mach.fitresult

function predict(
        model::RSirusRegressor,
        fitresult::RObject,
        Xnew
    )
    if !Tables.istable(Xnew)
        error("Expected a Table but got $(typeof(Xnew))")
    end
    df = DataFrame(Xnew)
    predictions = R"""
        sirus.predict($fitresult, $df)
    """
    return rcopy(predictions)
end

predict(mach, X)
e = _evaluate(model, X, y; measure=rsq)
@test 0.6 < _score(e)

MMI.@mlj_model mutable struct RSirusClassifier <: Probabilistic
    max_depth::Int=2
    max_rules::Int=10
end

function fit(
        model::RSirusClassifier,
        verbosity::Int,
        X,
        y::CategoricalArray
    )
    # Based on MLJXGBoostInterface.
    a_target_element = y[1]
    @assert a_target_element isa CategoricalValue

    if !Tables.istable(X)
        error("Expected a Table but got $(typeof(Xnew))")
    end
    df = DataFrame(X)
    outcomes = get.(y)
    fitted_model = R"""
        fitted.model <- sirus.fit(
            $df,
            $outcomes,
            type="classif",
            num.rule=$(model.max_rules),
            p0=NULL,
            num.rule.max =$(model.max_rules),
            q=4,
            max.depth=$(model.max_depth),
            num.trees=NULL,
            num.threads=1,
            verbose=FALSE,
            seed=1
        )
        # print(sirus.print(fitted.model))
        fitted.model
    """
    fitresult = (fitted_model, a_target_element)
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

y = categorical(rand(_rng(), [0, 1], n))

model = RSirusClassifier()
mach = machine(model, X, y, verbosity)
fit!(mach; verbosity)

function predict(
        model::RSirusClassifier,
        fitresult::Tuple{RObject, CategoricalValue},
        Xnew
    )
    fitted_model, a_target_element = fitresult
    if !Tables.istable(Xnew)
        error("Expected a Table but got $(typeof(Xnew))")
    end
    df = DataFrame(Xnew)
    rpredictions = R"""
        sirus.predict($fitted_model, $df)
    """
    classes = MMI.classes(a_target_element)
    predictions = rcopy(rpredictions)
    augment = ndims(predictions) == 1
    @show classes
    @show predictions
    @show augment
    return UnivariateFinite(classes, predictions; augment)
end

predict(mach, X)

e = _evaluate(model, X, y; measure=auc)
@test 0.5 < _score(e)

# Looks like sirus does not support multiclass classification.
# y = categorical(rand(_rng(), [0, 1, 2], n))
# _evaluate(model, X, y; measure=accuracy)
