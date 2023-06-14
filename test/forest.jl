X = [1 2;
     3 4]
y = [1, 2]

y_view = Vector{eltype(y)}(undef, length(y))
feature = 1
@test collect(ST._view_y!(y_view, X[:, feature], [1 2], <, 2)) == [1]
@test collect(ST._view_y!(y_view, X[:, feature], [1 2], >, 2)) == [2]

nothing
