n = 200
p = 40
rng = StableRNG(1)
X, y = make_blobs(n, p; centers=2, rng, shuffle=true)

_score(e::MLJBase.PerformanceEvaluation) = only(e.measurement)
function _evaluate(model; X=X, y=y)
    rng = StableRNG(1)
    resampling = CV(; shuffle=true, rng)
    evaluate(model, X, y; verbosity=0, resampling, measure=auc)
end

e = _evaluate(LGBMClassifier(; max_depth=2))
println("_evaluate(LGBMClassifier(...)) AUC: ", e)
@test 0.95 < _score(e)

rng = StableRNG(1)
model = StableForestClassifier(; rng)
mach = machine(model, X, y)
fit!(mach; verbosity=0)

preds = predict(mach)
println("StableForestClassifier(...) AUC: ", auc(preds, y))
@test 0.95 < auc(preds, y)

e = _evaluate(StableForestClassifier(; rng, n_trees=50))
println("StableForestClassifier AUC: ", e)
@test 0.95 < _score(e)

rng = StableRNG(1)
rulesmodel = StableRulesClassifier(; rng, p0=0.001, n_trees=50)
rulesmach = machine(rulesmodel, X, y)
fit!(rulesmach; verbosity=0)
preds = predict(rulesmach)

println("StableRulesClassifier AUC: ", auc(preds, y))
@test 0.95 < auc(preds, y)

rng = StableRNG(1)
e = _evaluate(StableRulesClassifier(; rng, p0=0.001, n_trees=50))
println("_evaluate(StableRulesClassifier(...)) AUC: ", e)
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
e = _evaluate(StableRulesClassifier(; rng, n_trees=1); X, y)
@test 0.7 < _score(e)

