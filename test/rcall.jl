using RCall

import MLJModelInterface:
    MLJModelInterface,
    fit,
    predict,
    metadata_model,
    metadata_pkg

using MLJModelInterface:
    MLJModelInterface,
    UnivariateFinite,
    Continuous,
    Count,
    Deterministic,
    Finite,
    Probabilistic,
    Table

const MMI = MLJModelInterface

# @rlibrary sirus

MMI.@mlj_model mutable struct RSirusRegressor <: Deterministic
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
    fitted_model = R"""
        fitted.model <- sirus.fit(
            $X,
            $y,
            type="auto",
            num.rule=10,
            p0=NULL,
            num.rule.max = 10,
            q=4,
            max.depth=2,
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
verbosity = 3
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

resampling = CV(; nfolds=3)
acceleration = MLJBase.CPUThreads()
e = evaluate(model, X, y; verbosity, acceleration, resampling, measure=rsq)
@test 0.6 < _score(e)
