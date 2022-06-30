n = 200
rng = StableRNG(1)
X, y = make_moons(n; rng, shuffle=true)

rng = StableRNG(1)
model = StableForestClassifier(; rng)
mach = machine(model, X, y)
fit!(mach)

preds = predict(mach)
@show round(accuracy(y, preds); digits=2)
@test 0.5 < accuracy(y, preds)

# evaluate(model, X, y; measure=auc)
