n = 200
p = 70
rng = StableRNG(1)
X, y = make_blobs(n, p; centers=2, rng, shuffle=true)

function _evaluate(rng, model)
    resampling = CV(; shuffle=true, rng)
    evaluate(model, X, y; verbosity=0, resampling, measure=auc)
end

rng = StableRNG(1)
@show _evaluate(rng, LGBMClassifier(; max_depth=2))

rng = StableRNG(1)
model = StableForestClassifier(; rng)
mach = machine(model, X, y)
fit!(mach; verbosity=0)

preds = predict(mach)
@show auc(preds, y)
@test 0.0 < auc(preds, y)

rng = StableRNG(1)
@show _evaluate(rng, StableForestClassifier(; rng))

rng = StableRNG(1)
rulesmodel = StableRulesClassifier(; rng)
rulesmach = machine(rulesmodel, X, y)
# fit!(rulesmach; verbosity=0)
# preds = predict(rulesmach)

# @show auc(preds, y)
# @test 0.0 < auc(preds, y)

# rng = StableRNG(1)
# @show _evaluate(rng, StableRulesClassifier(; rng))
