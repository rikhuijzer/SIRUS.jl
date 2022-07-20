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
        "haberman" => let
            df = haberman()
            X = MLJBase.table(MLJBase.matrix(df[:, Not(:survival)]))
            y = df.survival
            (X, y)
        end,
        "boston" => boston()
    )

function _score(e::PerformanceEvaluation)
    return round(only(e.measurement); sigdigits=2)
end

function _evaluate(model, X, y, nfolds=10)
    resampling = CV(; nfolds, shuffle=true, rng=_rng())
    acceleration = MLJBase.CPUThreads()
    evaluate(model, X, y; acceleration, verbosity=0, resampling, measure=auc)
end

results = DataFrame(;
        Dataset=String[],
        Model=String[],
        Hyperparameters=String[],
        nfolds=Int[],
        AUC=Float64[],
        se=Float64[]
    )

_filter_rng(hyper::NamedTuple) = Base.structdiff(hyper, (; rng=:foo))
_pretty_name(modeltype) = last(split(string(modeltype), '.'))
_hyper2str(hyper::NamedTuple) = hyper == (;) ? "(;)" : string(hyper)::String

function _evaluate!(
        results::DataFrame,
        dataset::String,
        modeltype::DataType,
        hyperparameters::NamedTuple=(; )
    )
    X, y = datasets[dataset]
    nfolds = 10
    model = modeltype(; hyperparameters...)
    e = _evaluate(model, X, y, nfolds)
    row = (;
        Dataset=dataset,
        Model=_pretty_name(modeltype),
        Hyperparameters=_hyper2str(_filter_rng(hyperparameters)),
        nfolds,
        AUC=_score(e),
        se=round(only(MLJBase._standard_errors(e)); digits=2)
    )
    push!(results, row)
    return e
end

function _evaluate_baseline!(results, dataset)
    _evaluate!(results, dataset, LGBMClassifier)
    e = _evaluate!(results, dataset, LGBMClassifier, (; max_depth=2))
    # _evaluate!(results, dataset, DecisionTreeClassifier, (; max_depth=2, rng=_rng()))
    return e
end

let
    e = _evaluate_baseline!(results, "blobs")
    @test 0.95 < _score(e)
end

X, y = datasets["blobs"]
model = StableForestClassifier(; rng=_rng())
mach = machine(model, X, y)
fit!(mach; verbosity=0)

preds = predict(mach)
@test 0.95 < auc(preds, y)

hyper= (; rng=_rng(), n_trees=50)
e = _evaluate(StableForestClassifier(; hyper...), X, y)
@test 0.95 < _score(e)
e2 = _evaluate(StableForestClassifier(; hyper...), X, y)
@test _score(e) == _score(e2)

rulesmodel = StableRulesClassifier(; n_trees=50, rng=_rng())
rulesmach = machine(rulesmodel, X, y)
fit!(rulesmach; verbosity=0)
preds = predict(rulesmach)
@test 0.95 < auc(preds, y)

let
    hyper = (; rng=_rng(), n_trees=50)
    e = _evaluate!(results, "blobs", StableRulesClassifier, hyper)
    @test 0.95 < _score(e)
end

n_trees = 40
e = _evaluate(StableRulesClassifier(; rng=_rng(), n_trees), X, y)
e2 = _evaluate(StableRulesClassifier(; rng=_rng(), n_trees), X, y)
@test _score(e) == _score(e2)
@test 0.7 < _score(e)

# e3 = _evaluate(StableRulesClassifier(; rng=_rng(), weight_penalty=0.0, n_trees); X, y)
# e4 = _evaluate(StableRulesClassifier(; rng=_rng(), weight_penalty=1.0, n_trees); X, y)
# @test _score(e3) != _score(e4)

let
    e = _evaluate_baseline!(results, "titanic")
    @test 0.83 < _score(e)
end

let
    hyper = (; rng=_rng(), n_trees=1_500)
    e = _evaluate!(results, "titanic", StableForestClassifier, hyper)
    @test 0.80 < _score(e)

    e = _evaluate!(results, "titanic", StableRulesClassifier, hyper)
    @test 0.80 < _score(e)
end

@testset "y as String" begin
    # https://github.com/rikhuijzer/StableTrees.jl/issues/5
    X = rand(10, 100)
    y = categorical(rand(["a", "b"], 10))
    model = StableForestClassifier()
    mach = machine(model, X, y; scitype_check_level=0)
    @test_throws ArgumentError fit!(mach)
end

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
    e = _evaluate_baseline!(results, "haberman")
    @test 0.64 < _score(e)
end

let
    hyper = (; rng=_rng(), n_trees=1_500)
    e = _evaluate!(results, "haberman", StableForestClassifier, hyper)
    @test 0.60 < _score(e)

    e = _evaluate!(results, "haberman", StableRulesClassifier, hyper)
    @test 0.60 < _score(e)
    fitresult = string(first(e.fitted_params_per_fold).fitresult)
    @test contains(fitresult, ":x1")
end

let
    e = _evaluate_baseline!(results, "boston")
end

let
    hyper = (; rng=_rng(), n_trees=1_500)
    e = _evaluate!(results, "boston", StableForestClassifier, hyper)

    e = _evaluate!(results, "boston", StableRulesClassifier, hyper)
end

pretty = rename(results, :se => "1.96*SE")
rename!(pretty, :nfolds => "`nfolds`")
print('\n' * repr(pretty) * "\n\n")

if haskey(ENV, "GITHUB_STEP_SUMMARY")
    job_summary = """
        ```
        $(repr(pretty))
        ```
        """
    path = ENV["GITHUB_STEP_SUMMARY"]
    open(path, "a") do io
        write(io, job_summary)
    end
end

nothing
