data = [0.0 2.5
        5.0 5.0
        0.0 0.0]

r1 = S.Rule(S.TreePath(" X[i, 1] < 1 "), [0.1], [0.0])
r2 = S.Rule(S.TreePath(" X[i, 2] < 2 "), [0.2], [0.0])

binary_feature_data = Float16[1 0; 0 0; 1 1]
@test S._binary_features([r1, r2], data) == binary_feature_data

y = Float16[0.5, 0, 1]

model = StableRulesClassifier()
@test S._estimate_coefficients(binary_feature_data, y, model) isa Vector
