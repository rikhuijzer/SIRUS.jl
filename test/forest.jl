@test ST._gini([1, 1], [1]) == 0.0
@test ST._gini([1, 0], [0, 1]) == 0.5
@test ST._gini([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) ≈ 0.8
@test ST._gini([1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]) ≈ 0.8

X = [1 2;
     3 4]
y = [1, 2]

feature = 1
@test collect(ST._view_y!(y_view, X[:, feature], [1 2], <, 2)) == [1]
@test collect(ST._view_y!(y_view, X[:, feature], [1 2], >, 2)) == [2]

@test ST._cutpoints([3, 1, 2], 2) == Float[1, 2]
@test ST._cutpoints(1:9, 3) == Float[3, 5, 7]
@test ST._cutpoints(1:4, 3) == Float[1, 2, 3]
@test ST._cutpoints([1, 3, 5, 7], 2) == Float[3, 5]

@test ST._cutpoints(X, 2) == [Float[1, 3], Float[2, 4]]
@test ST._cutpoints([3 4; 1 5; 2 6], 2) == [Float[1, 2], Float[4, 5]]

let
    X = [1 1;
         1 3]
    classes = unique(y)
    colnames = ["A", "B"]
    cutpoints = ST._cutpoints(X, 2)
    splitpoint = ST._split(StableRNG(1), X, y, classes, colnames, cutpoints)
    # Obviously, feature (column) 2 is more informative to split on than feature 1.
    @test splitpoint.feature == 2
    @test splitpoint.feature_name == "B"
    # Given that the split does < and ≥, then 3 is the best place since it separates 1 (left) and 3 (right).
    @test splitpoint.value == Float(3)
end

let
    X = [1 2; # 1
         3 4] # 2
    y = [1, 2]
    classes = y
    mask = Vector{Bool}(undef, length(y))
    node = ST._tree!(_rng(), mask, X, y, classes; min_data_in_leaf=1, q=2)
    # @test node.splitpoint == ST.SplitPoint(1, Float(3))
    # @test node.left.probabilities == [1.0, 0.0]
    # @test node.right.probabilities == [0.0, 1.0]
end

n = 200
p = 70
X, y = make_blobs(n, p; centers=2, rng=_rng(), shuffle=true)
colnames = ST._colnames(X)

n_subfeatures = 0
max_depth = 2
data = Tables.matrix(X)
dtree = DecisionTree.build_tree(unwrap.(y), data, n_subfeatures, max_depth; rng=_rng())
dpreds = DecisionTree.apply_tree(dtree, data)
@test 0.95 < accuracy(dpreds, y)

function _binary_accuracy(stree::ST.Node, classes, data, y)
    spreds = ST._predict(stree, data)
    binary = [x[1] < 0.5 ? classes[2] : classes[1] for x in spreds]
    return accuracy(binary, y)
end

classes = ST._classes(y)
mask = Vector{Bool}(undef, length(y))
stree = ST._tree!(_rng(), mask, data, y, classes, min_data_in_leaf=1, q=10)
@test 0.95 < _binary_accuracy(stree, classes, data, y)

@testset "data_subset" begin
    n_features = round(Int, sqrt(p))
    n_samples = round(Int, n/2)
    cols = rand(_rng(), 1:ST._p(data), n_features)
    rows = rand(_rng(), 1:length(y), n_samples)
    _data = view(data, rows, cols)
    _y = view(y, rows)

    dtree = DecisionTree.build_tree(unwrap.(_y), _data, n_subfeatures, max_depth; rng=_rng())
    dpreds = DecisionTree.apply_tree(dtree, _data)
    @test 0.95 < accuracy(dpreds, _y)

    mask = Vector{Bool}(undef, length(_y))
    stree = ST._tree!(_rng(), mask, _data, _y, classes, q=10)
    @test 0.95 < _binary_accuracy(stree, classes, _data, _y)
end

dforest = let
    n_subfeatures = -1
    n_trees = 10
    partial_sampling = 0.7
    max_depth = 2
    DecisionTree.build_forest(unwrap.(y), data, n_subfeatures, n_trees, partial_sampling, max_depth; rng=_rng())
end
# DecisionTree.print_tree.(dforest.trees);

sforest = ST._forest(_rng(), data, y, colnames; n_trees=10, max_depth=2)

@testset "max_depth is adhered to" begin
    some_children(forest) = ST.AbstractTrees.children(forest.trees[1])

    # ST.print_tree.(sforest.trees);
    @test !(all(child -> child isa ST.Leaf, some_children(sforest)))

    undeep_forest = ST._forest(_rng(), data, y, colnames; n_trees=10, max_depth=1)
    @test all(child -> child isa ST.Leaf, some_children(undeep_forest))
end

@testset "trees in forest are capable" begin
    dtree_accuracies = [accuracy(DecisionTree.apply_tree(tree, data), y) for tree in dforest.trees]
    @test all(>(0.95), dtree_accuracies)

    stree_accuracies = [_binary_accuracy(tree, classes, data, y) for tree in sforest.trees]
    # @test all(>(0.95), stree_accuracies)
end

fpreds = DecisionTree.apply_forest(dforest, data)
@show accuracy(fpreds, y)
@test 0.95 < accuracy(fpreds, y)

sfpreds = ST._predict(sforest, data)
@show accuracy(mode.(sfpreds), y)
@test 0.95 < accuracy(mode.(sfpreds), y)

empty_forest = ST.StableForest(Union{ST.Leaf, ST.Node}[], [1])
@test_throws AssertionError ST._predict(empty_forest, data)

nothing
