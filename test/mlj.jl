@testset "generic interface tests" begin
    data = MLJTestInterface.make_binary()
    kwargs = (
        mod = @__MODULE__,
        verbosity = 0,
        throw = false
    )
    failures, summary = MLJTestInterface.test([StableRulesClassifier], data...; kwargs...)
    @test isempty(failures)

    data = MLJTestInterface.make_multiclass()
    failures, summary = MLJTestInterface.test([StableRulesClassifier], data...; kwargs...)
    @test isempty(failures)

    data = MLJTestInterface.make_regression()
    failures, summary = MLJTestInterface.test([StableRulesRegressor], data...; kwargs...)
    @test isempty(failures)
end

@testset "keep classes as integers" begin
    _X, _y = datasets["haberman"]
    @test _y isa CategoricalVector{<:Int}
    model = StableRulesClassifier(; max_depth=2, max_rules=8, n_trees=10, rng=_rng())
    mach = machine(model, _X, _y)
    fit!(mach)
    classes = mach.fitresult.classes
    @test classes isa Vector{<:Int}
end

X, y = datasets["blobs"]
model = StableForestClassifier(; rng=_rng())
mach = machine(model, X, y)
fit!(mach; verbosity=0)

preds = predict(mach)
@test 0.95 < auc(preds, y)

hyper = (; rng=_rng(), n_trees=50)
e = _evaluate(StableForestClassifier(; hyper...), X, y)
@test 0.95 < _score(e)
e2 = _evaluate(StableForestClassifier(; hyper...), X, y)
@test _score(e) == _score(e2)

rulesmodel = StableRulesClassifier(; n_trees=50, rng=_rng())
rulesmach = machine(rulesmodel, X, y)
fit!(rulesmach; verbosity=0)
preds = predict(rulesmach)
@test 0.95 < auc(preds, y)

hyper = (; rng=_rng(), n_trees=1_000, max_depth=2, max_rules=10)
e = _evaluate(StableRulesClassifier(; hyper...), X, y)
@test 0.95 < _score(e)

n_trees = 40
e = _evaluate(StableRulesClassifier(; rng=_rng(), n_trees), X, y)
e2 = _evaluate(StableRulesClassifier(; rng=_rng(), n_trees), X, y)
@test _score(e) == _score(e2)
@test 0.7 < _score(e)
@testset "feature names" begin
    rng = _rng()
    n = 50
    X = DataFrame(; AAA=rand(rng, n), AAB=rand(rng, n))
    y = categorical(rand(rng, 1:2, n))
    model = StableRulesClassifier(; n_trees=50)
    mach = machine(model, X, y)
    fit!(mach)
    @test contains(repr(mach.fitresult), "AAB")
end

let
    data = "haberman"
    @info "Evaluating $data"
    hyper = (; rng=_rng())
    _evaluate!(results, data, DecisionTreeClassifier)

    hyper = (;)
    _evaluate!(results, data, LogisticClassifier, hyper)

    hyper = (;)
    acceleration = MLJBase.CPU1()
    e = _evaluate!(results, data, XGBoostClassifier, hyper; acceleration)

    hyper = (; max_depth=2)
    e = _evaluate!(results, data, XGBoostClassifier, hyper; acceleration)

    hyper = (; rng=_rng(), max_depth=2)
    e = _evaluate!(results, data, StableForestClassifier, hyper)
    @test 0.60 < _score(e)

    hyper = (; rng=_rng(), max_depth=2, max_rules=30)
    e = _evaluate!(results, data, StableRulesClassifier, hyper)
    @test 0.60 < _score(e)
    fitresult = string(first(e.fitted_params_per_fold).fitresult)
    @test contains(fitresult, ":x1")

    hyper = (; rng=_rng(), max_depth=2, max_rules=10)
    e = _evaluate!(results, data, StableRulesClassifier, hyper)
    @test 0.60 < _score(e)

    if get(ENV, "CAN_RUN_R_SIRUS", "false") == "true"
        hyper = (; max_depth=2, max_rules=10)
        e = _evaluate!(results, data, RSirusClassifier, hyper; acceleration)
    end
end

let
    data = "titanic"
    @info "Evaluating $data"
    hyper = (; rng=_rng())
    e = _evaluate!(results, data, DecisionTreeClassifier, hyper)

    hyper = (;)
    e = _evaluate!(results, data, LogisticClassifier, hyper)

    hyper = (;)
    acceleration = MLJBase.CPU1()
    e = _evaluate!(results, data, XGBoostClassifier, hyper; acceleration)

    hyper = (; max_depth=2)
    e = _evaluate!(results, data, XGBoostClassifier, hyper; acceleration)

    hyper = (; rng=_rng(), max_depth=2)
    e = _evaluate!(results, data, StableForestClassifier, hyper)
    @test 0.80 < _score(e)

    hyper = (; rng=_rng(), max_depth=2, max_rules=30)
    e = _evaluate!(results, data, StableRulesClassifier, hyper)
    @test 0.79 < _score(e)

    hyper = (; rng=_rng(), max_depth=2, max_rules=10)
    e = _evaluate!(results, data, StableRulesClassifier, hyper)
    @test 0.79 < _score(e)

    if get(ENV, "CAN_RUN_R_SIRUS", "false") == "true"
        hyper = (; max_depth=2, max_rules=10)
        e = _evaluate!(results, data, RSirusClassifier, hyper; acceleration)
    end
end

let
    data = "cancer"
    @info "Evaluating $data"
    measure = auc

    hyper = (; rng=_rng())
    e = _evaluate!(results, data, DecisionTreeClassifier, hyper; measure)

    hyper = (;)
    e = _evaluate!(results, data, MultinomialClassifier, hyper; measure)

    hyper = (;)
    acceleration = MLJBase.CPU1()
    e = _evaluate!(results, data, XGBoostClassifier, hyper; measure, acceleration)

    hyper = (; max_depth=2)
    e = _evaluate!(results, data, XGBoostClassifier, hyper; measure, acceleration)

    hyper = (; rng=_rng(), max_depth=2)
    e = _evaluate!(results, data, StableForestClassifier, hyper; measure)

    hyper = (; rng=_rng(), max_depth=2, max_rules=30)
    e = _evaluate!(results, data, StableRulesClassifier, hyper; measure)

    hyper = (; rng=_rng(), max_depth=2, max_rules=10)
    e = _evaluate!(results, data, StableRulesClassifier, hyper; measure)

    if get(ENV, "CAN_RUN_R_SIRUS", "false") == "true"
        hyper = (; max_depth=2, max_rules=10)
        e = _evaluate!(results, data, RSirusClassifier, hyper; measure, acceleration)
    end
end

let
    data = "diabetes"
    @info "Evaluating $data"
    hyper = (; rng=_rng())
    e = _evaluate!(results, data, DecisionTreeClassifier, hyper)

    hyper = (;)
    e = _evaluate!(results, data, LogisticClassifier, hyper)

    hyper = (;)
    acceleration = MLJBase.CPU1()
    e = _evaluate!(results, data, XGBoostClassifier, hyper; acceleration)

    hyper = (; max_depth=2)
    e = _evaluate!(results, data, XGBoostClassifier, hyper; acceleration)

    hyper = (; rng=_rng(), max_depth=2)
    e = _evaluate!(results, data, StableForestClassifier, hyper)

    hyper = (; rng=_rng(), max_depth=2, max_rules=30)
    e = _evaluate!(results, data, StableRulesClassifier, hyper)

    hyper = (; rng=_rng(), max_depth=2, max_rules=10)
    e = _evaluate!(results, data, StableRulesClassifier, hyper)

    if get(ENV, "CAN_RUN_R_SIRUS", "false") == "true"
        hyper = (; max_depth=2, max_rules=10)
        e = _evaluate!(results, data, RSirusClassifier, hyper; acceleration)
    end
end

let
    data = "iris"
    @info "Evaluating $data"
    measure = accuracy

    hyper = (; rng=_rng())
    e = _evaluate!(results, data, DecisionTreeClassifier, hyper; measure)

    hyper = (;)
    e = _evaluate!(results, data, MultinomialClassifier, hyper; measure)

    hyper = (;)
    acceleration = MLJBase.CPU1()
    e = _evaluate!(results, data, XGBoostClassifier, hyper; measure, acceleration)

    hyper = (; max_depth=2)
    e = _evaluate!(results, data, XGBoostClassifier, hyper; measure, acceleration)

    hyper = (; rng=_rng(), max_depth=2)
    e = _evaluate!(results, data, StableForestClassifier, hyper; measure)
    @test 0.90 < _score(e)

    lambda = 0.01
    hyper = (; rng=_rng(), max_depth=2, max_rules=30, lambda)
    e = _evaluate!(results, data, StableRulesClassifier, hyper; measure)

    hyper = (; rng=_rng(), max_depth=2, max_rules=10, lambda)
    e = _evaluate!(results, data, StableRulesClassifier, hyper; measure)
    @test 0.62 < _score(e)

    # R sirus doesn't appear to support multiclass classification.
end

rulesmodel = StableRulesRegressor(; max_depth=2, max_rules=30, rng=_rng())
X, y = datasets["boston"]
rulesmach = machine(rulesmodel, X, y)
fit!(rulesmach; verbosity=0)
preds = predict(rulesmach)
@test preds isa Vector{Float64}
# @show rsq(preds, y)
# @test 0.95 < rsq(preds, y)

let
    data = "boston"
    @info "Evaluating $data"
    measure = rsq
    hyper = (; rng=_rng())
    elgbm = _evaluate!(results, data, DecisionTreeRegressor, hyper; measure)

    hyper = (;)
    _evaluate!(results, data, LinearRegressor, hyper; measure)

    hyper = (;)
    acceleration = MLJBase.CPU1()
    e = _evaluate!(results, data, XGBoostRegressor, hyper; measure, acceleration)

    hyper = (; max_depth=2)
    ex = _evaluate!(results, data, XGBoostRegressor, hyper; measure, acceleration)

    hyper = (; max_depth=2, rng=_rng())
    ef = _evaluate!(results, data, StableForestRegressor, hyper; measure)

    @test 0.62 < _score(ex)

    hyper = (; rng=_rng(), max_depth=2, max_rules=30)
    er = _evaluate!(results, data, StableRulesRegressor, hyper; measure)

    hyper = (; rng=_rng(), max_depth=2, max_rules=10)
    er = _evaluate!(results, data, StableRulesRegressor, hyper; measure)
    @test 0.55 < _score(er)

    if get(ENV, "CAN_RUN_R_SIRUS", "false") == "true"
        hyper = (; max_depth=2, max_rules=10)
        _evaluate!(results, data, RSirusRegressor, hyper; measure, acceleration)
    end
end

emr = let
    measure = rsq
    data = "make_regression"
    @info "Evaluating $data"
    hyper = (; rng=_rng())
    _evaluate!(results, data, DecisionTreeRegressor, hyper; measure)

    hyper = (;)
    e = _evaluate!(results, data, LinearRegressor, hyper; measure)

    hyper = (;)
    acceleration = MLJBase.CPU1()
    e = _evaluate!(results, data, XGBoostRegressor, hyper; measure, acceleration)

    hyper = (; max_depth=2)
    e = _evaluate!(results, data, XGBoostRegressor, hyper; measure, acceleration)

    hyper = (; max_depth=2, rng=_rng())
    _evaluate!(results, data, StableForestRegressor, hyper; measure)

    # With ridge regression, a high lambda makes all coefficients very small.
    # This makes sense for the regression task since the rule-based algorithm
    # cannot fit a straight line well. In other words, many small rules have
    # to be fitted and work together.
    lambda = 100
    q = 20
    hyper = (; rng=_rng(), max_depth=2, max_rules=30, lambda, q)
    _evaluate!(results, data, StableRulesRegressor, hyper; measure)

    hyper = (; rng=_rng(), max_depth=2, max_rules=10, lambda, q)
    er = _evaluate!(results, data, StableRulesRegressor, hyper; measure)
    @test 0.50 < _score(er)

    if get(ENV, "CAN_RUN_R_SIRUS", "false") == "true"
        hyper = (; max_depth=2, max_rules=10)
        _evaluate!(results, data, RSirusRegressor, hyper; measure, acceleration)
    end
end

pretty = rename(results, :se => "1.96*SE")
# rename!(pretty, :nfolds => "`nfolds`")
print('\n' * repr(pretty) * "\n\n")

step_summary_path = get(ENV, "GITHUB_STEP_SUMMARY", "nothing")
if step_summary_path != "nothing"
    job_summary = """
        ```
        $(repr(pretty))

        with Julia version $VERSION
        ```
        """
    write(step_summary_path, job_summary)
end

nothing
