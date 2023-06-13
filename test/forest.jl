X = [1 2;
     3 4]
y = [1, 2]

y_view = Vector{eltype(y)}(undef, length(y))
feature = 1
@test collect(ST._view_y!(y_view, X[:, feature], [1 2], <, 2)) == [1]
@test collect(ST._view_y!(y_view, X[:, feature], [1 2], >, 2)) == [2]

let
    X = [1 1;
         1 3]
    classes = unique(y)
    colnames = ["A", "B"]
    cp = cutpoints(X, 2)
    max_split_candidates::Int = SIRUS.nfeatures(X)
    splitpoint = ST._split(StableRNG(1), X, y, classes, colnames, cp; max_split_candidates)
    # Obviously, feature (column) 2 is more informative to split on than feature 1.
    @test splitpoint.feature == 2
    @test splitpoint.feature_name == "B"
    # Given that the split does < and ≥, then 3 is the best place since it separates 1 (left) and 3 (right).
    @test splitpoint.value == Float32(3)
end

nothing
