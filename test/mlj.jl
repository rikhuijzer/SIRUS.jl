n = 200
X, y = make_moons(n)

rng = StableRNG(1)
model = StableForestClassifier(; rng)
mach = machine(model, X, y)
fit!(mach)

preds = predict(mach)
@test 0.6 < accuracy(y, preds)
