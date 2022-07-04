@test ST._gini([1, 1]) == 0.0
@test ST._gini([1, 0]) == 0.5
@test ST._gini([1, 2, 3, 4, 5]) ≈ 0.8

X = [1 2;
     3 4]
y = [1, 2]

feature = 1
@test collect(ST._view_y(X, [1 2], feature, <, 2)) == [1]
@test collect(ST._view_y(X, [1 2], feature, >, 2)) == [2]

@test ST._cutpoints([3, 1, 2], 2) == Float[1, 3]
@test ST._cutpoints(1:9, 3) == Float[1, 5, 9]
@test ST._cutpoints(1:4, 3) == Float[1, 2, 4]
@test ST._cutpoints([1, 3, 5, 7], 4) == Float[1, 3, 5, 7]

@test ST._cutpoints(X, 2) == Float[1 2; 3 4]
@test ST._cutpoints([3 4; 1 5; 2 6], 2) == Float[1 4; 3 6]

let
    X = [1 1;
         1 3]
    classes = unique(y)
    cutpoints = Float.(X)
    splitpoint = ST._split(X, y, classes, cutpoints)
    # Obviously, feature (column) 2 is more informative to split on than feature 1.
    @test splitpoint.feature == 2
    # Given that the split does < and ≥, then 3 is the best place since it separates 1 (left) and 3 (right).
    @test splitpoint.value == Float(3)
end

let
    X = [1 2; # 1
         3 4] # 2
    y = [1, 2]
    classes = y
    node = ST._tree(X, y, classes; min_data_in_leaf=1, q=2)
    # @test node.splitpoint == ST.SplitPoint(1, Float(3))
    # @test node.left.probabilities == [1.0, 0.0]
    # @test node.right.probabilities == [0.0, 1.0]
end

n = 200
p = 70
rng = StableRNG(1)
X, y = make_blobs(n, p; centers=2, rng, shuffle=true)

let
    n_subfeatures = 0
    max_depth = 2
    data = Tables.matrix(X)
    dtree = DecisionTree.build_tree(unwrap.(y), data, n_subfeatures, max_depth)
    dpreds = DecisionTree.apply_tree(tree, data)
    @test 0.95 < accuracy(dpreds, y)

    classes = unique(y)
    stree = ST._tree(data, y, classes, min_data_in_leaf=1, q=10)
    spreds = ST._predict(stree, data)
    spreds = [x[1] < 0.5 ? classes[2] : classes[1] for x in spreds]
    @test 0.95 < accuracy(spreds, y)
end
