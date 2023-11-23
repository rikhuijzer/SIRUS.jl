function _haberman_data()
    df = haberman()
    X = MLJBase.table(MLJBase.matrix(df[:, Not(:survival)]))
    y = categorical(df.survival)
    (X, y)
end

X, y = _haberman_data()

mach = let
    classifier = StableRulesClassifier(; max_depth=1, max_rules=8, n_trees=1000, rng=_rng())
    mach = machine(classifier, X, y)
    fit!(mach)
end

model = mach.fitresult::StableRules
# StableRules model with 8 rules:
#  if X[i, :x3] < 8.0 then 0.084 else 0.03 +
#  if X[i, :x3] < 14.0 then 0.147 else 0.098 +
#  if X[i, :x3] < 2.0 then 0.073 else 0.047 +
#  if X[i, :x3] < 4.0 then 0.079 else 0.048 +
#  if X[i, :x3] < 1.0 then 0.076 else 0.06 +
#  if X[i, :x2] < 1959.0 then 0.006 else 0.008 +
#  if X[i, :x1] < 38.0 then 0.029 else 0.024 +
#  if X[i, :x1] < 42.0 then 0.052 else 0.043
# and 2 classes: [0, 1].
# Note: showing only the probability for class 1 since class 0 has probability 1 - p.

importance = feature_importance(model, "x1")
# Based on the numbers above.
expected = ((0.029 - 0.024) + (0.052 - 0.043))
@test importance ≈ expected atol=0.01

@test feature_importance([model, model], "x1") ≈ expected atol=0.01
@test only(feature_importances(model, ["x1"])).importance ≈ expected atol=0.01
