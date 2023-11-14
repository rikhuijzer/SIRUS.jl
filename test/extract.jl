function _haberman_data()
    df = haberman()
    X = MLJBase.table(MLJBase.matrix(df[:, Not(:survival)]))
    y = categorical(df.survival)
    (X, y)
end

X, y = _haberman_data()

classifier = StableRulesClassifier(; max_depth=2, max_rules=8, n_trees=1000, rng=_rng())
mach = machine(classifier, X, y)
fit!(mach)

model = mach.fitresult::StableRules

importance = SIRUS.feature_importance(model, "x1")
