data = [0.0 2.5
        5.0 5.0
        0.0 0.0]

r1 = S.Rule(S.Clause(" X[i, 1] < 1 "), [0.1], [0.0])
r2 = S.Rule(S.Clause(" X[i, 2] < 2 "), [0.2], [0.0])

binary_feature_data = Float16[1 0;
                              0 0;
                              1 1]
@test S._binary_features([r1, r2], data) == binary_feature_data

y = Float16[0.5, 0, 1]

model = StableRulesClassifier()
algo = S.Classification()
@test S._estimate_coefficients!(algo, binary_feature_data, copy(y), model) isa Vector

@test SIRUS._normalize!([1.0, 2.0, 3.0]) == [0.0, 0.5, 1.0]

nothing
