n = 200
p = 40
X, y = make_blobs(n, p; centers=2, rng=_rng(), shuffle=true)

function _score(e::PerformanceEvaluation)
    return round(only(e.measurement); sigdigits=2)
end

function _evaluate(model; X=X, y=y)
    resampling = CV(; nfolds=14, shuffle=true, rng=_rng())
    acceleration = MLJBase.CPUThreads()
    evaluate(model, X, y; acceleration, verbosity=0, resampling, measure=auc)
end

e = _evaluate(LGBMClassifier(; max_depth=2))
println("_evaluate(LGBMClassifier) AUC: ", _score(e))
@test 0.95 < _score(e)

model = StableForestClassifier(; rng=_rng())
mach = machine(model, X, y)
fit!(mach; verbosity=0)

preds = predict(mach)
println("StableForestClassifier AUC: ", auc(preds, y))
@test 0.95 < auc(preds, y)

e = _evaluate(StableForestClassifier(; rng=_rng(), n_trees=50))
println("StableForestClassifier AUC: ", _score(e))
@test 0.95 < _score(e)
e2 = _evaluate(StableForestClassifier(; rng=_rng(), n_trees=50))
@test _score(e) == _score(e2)

rulesmodel = StableRulesClassifier(; rng=_rng(), n_trees=50)
rulesmach = machine(rulesmodel, X, y)
fit!(rulesmach; verbosity=0)
preds = predict(rulesmach)

println("StableRulesClassifier AUC: ", auc(preds, y))
@test 0.95 < auc(preds, y)

e = _evaluate(StableRulesClassifier(; rng=_rng(), n_trees=50))
println("_evaluate(StableRulesClassifier) AUC: ", _score(e))
@test 0.95 < _score(e)

titanic = Titanic()
X, y = let
    df = titanic.features
    F = [:Pclass, :Sex, :Age, :SibSp, :Parch, :Fare, :Embarked]
    sub = select(df, F...)
    sub[!, :y] = categorical(titanic.targets[:, 1])
    sub[!, :Sex] = ifelse.(sub.Sex .== "male", 1, 0)
    dropmissing!(sub)
    embarked2int(x) = x == "S" ? 1 : x == "C" ? 2 : 3
    sub[!, :Embarked] = embarked2int.(sub.Embarked)
    (select(sub, Not(:y)), sub.y)
end
n_trees = 40
e = _evaluate(StableRulesClassifier(; rng=_rng(), n_trees); X, y)
e2 = _evaluate(StableRulesClassifier(; rng=_rng(), n_trees); X, y)
@test _score(e) == _score(e2)
@test 0.7 < _score(e)

# e3 = _evaluate(StableRulesClassifier(; rng=_rng(), weight_penalty=0.0, n_trees); X, y)
# e4 = _evaluate(StableRulesClassifier(; rng=_rng(), weight_penalty=1.0, n_trees); X, y)
# @test _score(e3) != _score(e4)

e = _evaluate(StableRulesClassifier(; rng=_rng(), n_trees=1500); X, y)
println("Titanic _evaluate(StableRulesClassifier) AUC: ", _score(e))
@test 0.80 < _score(e)

Xt = MLJBase.table(MLJBase.matrix(X))
le = _evaluate(LGBMClassifier(; max_depth=10); X=Xt, y)
println("Titanic _evaluate(LGBMClassifier) AUC: ", _score(le))
@test 0.83 < _score(le)

