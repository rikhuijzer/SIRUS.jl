@test ST.gini([1, 1, 1], [1]) == Float(0.0)

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

_mode = ST.MLJImplementation._mode
@test _mode([1, 2, 3, 4, 4]) == 4
@test _mode([1, 2, 3, 1]) == 1
@test _mode([1]) == 1
@test _mode([1, 2]) in [1, 2]

c = categorical([1, 2, 2])
# Otherwise the type of y isn't the same as the type for predictions for y.
@test _mode(c) isa CategoricalValue
@test _mode(c) == c[2]

let
    X = [1 2; # 1
         3 4] # 2
    y = [1, 2]
    classes = y
    node = ST._tree(X, y, classes; min_data_in_leaf=1, q=2)
    @test node.splitpoint == ST.SplitPoint(1, Float(3))
    @test node.left.probabilities == Float[1, 0]
    @test node.right.probabilities == Float[0, 1]
end

let
    X = Float64[1 2; # 1
                3 2; # 2
                5 2; # 3
                7 2] # 4
    y = [1, 2, 3, 4]
    node = ST._tree(X, y; min_data_in_leaf=1, q=3)
    # This looks a bit weird at first sight but makes sense due to greedy recursive binary splitting.
    # When retaining only one element, the gini index of one side is 0 because the node predicts one class perfectly.
    # See "An introduction to Statistical Learning" for details.
    # So decision trees are not optimized to be well balanced.
    @test node.splitpoint == ST.SplitPoint(1, Float(3))
    # @test node.right.majority == 4
    # @test node.right.n == 1
    # @test node.left.splitpoint == ST.SplitPoint(1, Float(5))
    # All datapoints are 3 here, 4 was already part of an earlier leaf.
    # @test node.left.right.majority == 3
    # @test node.left.right.n == 1

    # @test ST._predict(node, [7, 2]) == 4
    # @test ST._predict(node, [5, 2]) == 3
end
