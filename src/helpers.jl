module Helpers

"Return the number of features `p` in a dataset `X`."
nfeatures(X::AbstractMatrix) = size(X, 2)
nfeatures(X) = length(Tables.columnnames(X))

view_feature(X::AbstractMatrix, feature::Int) = view(X, :, feature)
view_feature(X, feature::Int) = view(X, feature)

end
