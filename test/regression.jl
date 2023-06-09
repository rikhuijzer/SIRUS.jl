X = [1.0 2.0;
     3.0 4.0]
y = [0.5, 0.6]

sforest = ST._forest(_rng(), data, y, colnames; n_trees=10, max_depth=2)
