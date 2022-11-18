data = [0.0 2.5
        5.0 5.0
        0.0 0.0]

r1 = ST.Rule(ST.TreePath(" X[i, 1] < 1 "), [0.1], [0.0])
r2 = ST.Rule(ST.TreePath(" X[i, 2] < 2 "), [0.2], [0.0])

binary_feature_data = BitMatrix([1 0; 0 0; 1 1])
@test ST._binary_features([r1, r2], data) == binary_feature_data

y = [0.5, 0, 1]

model = StableRulesClassifier()
@test ST._estimate_coefficients(binary_feature_data, y, model) isa Vector{Float64}
