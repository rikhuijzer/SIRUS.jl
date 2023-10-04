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
    
    sfit = R"""
        sirus.fit(
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
            verbose=TRUE,
            seed=1
        )
    """
end

model = RSirusRegressor()
mach = machine(model, X, y)
fit!(mach)
