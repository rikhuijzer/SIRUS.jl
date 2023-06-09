X = [1.0 2.0;
     3.0 4.0]
y = [0.5, 0.6]

output_type = :regression

sforest = ST._forest(_rng(), output_type, data, y, colnames; n_trees=10, max_depth=2)
