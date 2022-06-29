import Base

using StableTrees
using Test

const ST = StableTrees
const Float = ST.Float

@test ST.gini([1, 1, 1], [1]) == Float(0.0)

X = [1 2;
     3 4]
y = [1, 2]

feature = 1
@test collect(ST._view_y(X, [1 2], feature, <, 2)) == [1]
@test collect(ST._view_y(X, [1 2], feature, >, 2)) == [2]

@test ST._cutpoints([3, 1, 2], 2) == Float[1, 3]
@test ST._cutpoints(1:10, 3) == Float[1, 5, 10]

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

@test ST._mode([1, 2, 3, 4, 4]) == 4
@test ST._mode([1, 2, 3, 1]) == 1
@test ST._mode([1]) == 1
@test ST._mode([1, 2]) in [1, 2]

function Base.:(==)(a::ST.SplitPoint, b::ST.SplitPoint)
    return a.feature == b.feature && a.value == b.value
end

function Base.:(==)(a::ST.Leaf, b::ST.Leaf)
    return a.majority == b.majority && a.values == b.values
end

let
    node = ST._tree([1 2; 3 3], [1, 2]; min_data_in_leaf=1, q=2)
    @test node.splitpoint == ST.SplitPoint(1, Float(3))
    let
        majority = 1
        data = [1]
        @test node.left == ST.Leaf{Int}(majority, data)
    end
    let
        majority = 2
        data = [2]
        @test node.right == ST.Leaf{Int}(majority, data)
    end
end
