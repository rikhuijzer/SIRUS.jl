@test SIRUS._rss([1, 2, 3]) == 2.0
@test SIRUS._rss(1:100) == 83325.0

algo = SIRUS.Regression()

X, y = boston()
colnames = SIRUS.colnames(X)

data = Tables.matrix(X)

d_preds = let
    n_subfeatures = 0
    max_depth = 2
    dtree = DecisionTree.build_tree(y, data, n_subfeatures, max_depth; rng=_rng())
    DecisionTree.apply_tree(dtree, data)
end

s_preds = let
    classes = []
    mask = Vector{Bool}(undef, length(y))
    stree = SIRUS._tree!(_rng(), algo, mask, data, y, classes, min_data_in_leaf=1, q=100)
    SIRUS._predict(stree, data)
end

# Note that this low performance is for one tree only.
@test 0.6 < rsq(d_preds, y)
@test rsq(d_preds, y) ≈ rsq(s_preds, y) atol=0.15

# TODO: test a forest.

nothing
