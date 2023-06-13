output_type = SIRUS.Classification()

X = [1 2;
     3 4]
y = [1, 2]

@test SIRUS._count_equal([1, 2, 3, 1], 1) == 2

@test SIRUS._gini([1, 1], [1]) == 0.0
@test SIRUS._gini([1, 0], [0, 1]) == 0.5
@test SIRUS._gini([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) ≈ 0.8
@test SIRUS._gini([1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]) ≈ 0.8

let
    X = [1 2; # 1
         3 4] # 2
    y = [1, 2]
    classes = y
    mask = Vector{Bool}(undef, length(y))
    node = SIRUS._tree!(_rng(), output_type, mask, X, y, classes; min_data_in_leaf=1, q=2)
    # @test node.splitpoint == SIRUS.SplitPoint(1, Float32(3))
    # @test node.left.probabilities == [1.0, 0.0]
    # @test node.right.probabilities == [0.0, 1.0]
end

n = 200
p = 70
X, y = make_blobs(n, p; centers=2, rng=_rng(), shuffle=true)
colnames = SIRUS.colnames(X)

n_subfeatures = 0
max_depth = 2
data = Tables.matrix(X)
dtree = DecisionTree.build_tree(unwrap.(y), data, n_subfeatures, max_depth; rng=_rng())
dpreds = DecisionTree.apply_tree(dtree, data)
@test 0.95 < accuracy(dpreds, y)

function _binary_accuracy(stree::SIRUS.Node, classes, data, y)
    spreds = SIRUS._predict(stree, data)
    binary = [x[1] < 0.5 ? classes[2] : classes[1] for x in spreds]
    return accuracy(binary, y)
end

classes = SIRUS._classes(y)
mask = Vector{Bool}(undef, length(y))
stree = SIRUS._tree!(_rng(), output_type, mask, data, y, classes, min_data_in_leaf=1, q=10)
@test 0.95 < _binary_accuracy(stree, classes, data, y)

@testset "data_subset" begin
    n_features = round(Int, sqrt(p))
    n_samples = round(Int, n/2)
    cols = rand(_rng(), 1:SIRUS.nfeatures(data), n_features)
    rows = rand(_rng(), 1:length(y), n_samples)
    _data = view(data, rows, cols)
    _y = view(y, rows)

    dtree = DecisionTree.build_tree(unwrap.(_y), _data, n_subfeatures, max_depth; rng=_rng())
    dpreds = DecisionTree.apply_tree(dtree, _data)
    @test 0.95 < accuracy(dpreds, _y)

    mask = Vector{Bool}(undef, length(_y))
    stree = SIRUS._tree!(_rng(), output_type, mask, _data, _y, classes, q=10)
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

sforest = SIRUS._forest(_rng(), output_type, data, y, colnames; n_trees=10, max_depth=2)

@testset "max_depth is adhered to" begin
    some_children(forest) = SIRUS.AbstractTrees.children(forest.trees[1])

    # SIRUS.print_tree.(sforest.trees);
    @test !(all(child -> child isa SIRUS.Leaf, some_children(sforest)))

    undeep_forest = SIRUS._forest(_rng(), output_type, data, y, colnames; n_trees=10, max_depth=1)
    @test all(child -> child isa SIRUS.Leaf, some_children(undeep_forest))
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

sfpreds = SIRUS._predict(sforest, data)
@show accuracy(mode.(sfpreds), y)
@test 0.95 < accuracy(mode.(sfpreds), y)

empty_forest = SIRUS.StableForest(Union{SIRUS.Leaf, SIRUS.Node}[], [1])
@test_throws AssertionError SIRUS._predict(empty_forest, data)

nothing
