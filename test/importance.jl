r1 = S.Rule(S.Clause(" X[i, 1] < 32000 "), [0.1], [0.4])
r2 = S.Rule(S.Clause(" X[i, 1] ≥ 32000 "), [0.3], [0.2])
r3 = S.Rule(S.Clause(" X[i, 2] < 8000 "), [0.1], [0.5])
w1 = 0.4
w2 = 0.3
w3 = 0.3

model = let
    rules = [r1, r2, r3]
    algo = SIRUS.Classification()
    classes = [0, 1]
    weights = Float16[w1, w2, w3]
    SIRUS.StableRules(rules, algo, classes, weights)
end
# StableRules model with 3 rules:
#  if X[i, 1] < 32000.0 then [0.04] else [0.16] +
#  if X[i, 1] ≥ 32000.0 then [0.09] else [0.06] +
#  if X[i, 2] < 8000.0 then [0.03] else [0.15]
# and 2 classes: [0, 1].
# Note: showing only the probability for class 1 since class 0 has probability 1 - p.

importance = feature_importance(model, "1")
# Based on the numbers above.
expected = w1 * (0.4 - 0.1) + w2 * (0.3 - 0.2)
@test importance ≈ expected atol=0.01

@test feature_importance([model, model], "1") ≈ expected atol=0.01
@test only(feature_importances(model, ["1"])).importance ≈ expected atol=0.01

importances = feature_importances([model], ["1", "2"])::Vector{<:NamedTuple}
@test length(importances) == 2
@test importances[1].feature_name == "1"
@test importances[1].importance ≈ expected atol=0.01
@test importances[2].feature_name == "2"

@test unpack_rule(r1) == (;
        feature=1,
        feature_name="1",
        splitval=32000.0,
        direction=:L,
        then=[0.1],
        otherwise=[0.4]
    )

@test unpack_model(model)[1] == (;
        weight=Float16(w1),
        feature=1,
        feature_name="1",
        splitval=32000.0,
        direction=:L,
        then=[0.1],
        otherwise=[0.4]
    )
