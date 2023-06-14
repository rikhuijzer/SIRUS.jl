@test SIRUS._rss([1, 2, 3]) == 2.0
@test SIRUS._rss(1:100) == 83325.0

X = [1.0 2.0;
     3.0 4.0]
y = [0.5, 0.6]

algo = SIRUS.Regression()

sforest = ST._forest(_rng(), algo, data, y, colnames; n_trees=10, max_depth=2)

nothing
