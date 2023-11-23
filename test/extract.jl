function _haberman_data()
    df = haberman()
    X = MLJBase.table(MLJBase.matrix(df[:, Not(:survival)]))
    y = categorical(df.survival)
    (X, y)
end

X, y = _haberman_data()

classifier = StableRulesClassifier(; max_depth=1, max_rules=8, n_trees=1000, rng=_rng())
mach = machine(classifier, X, y)
fit!(mach)

model = mach.fitresult::StableRules

importance = feature_importance(model, "x1")
# Based on the numbers that are printed in the following lines:
# if X[i, :x1] < 38.0 then 0.029 else 0.024 +
# if X[i, :x1] < 42.0 then 0.052 else 0.043
expected = ((0.029 - 0.024) + (0.052 - 0.043))
@test importance ≈ expected atol=0.01

@test feature_importance([model, model], "x1") ≈ expected atol=0.01
