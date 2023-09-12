module Helpers

using Tables: Tables

"Return `names(X)` if defined for `X` and string numbers otherwise."
function colnames(X)::Vector{String}
    fallback() = string.(1:nfeatures(X))
    try
        names = collect(string.(Tables.columnnames(X)))
        if isempty(names)
            return fallback()
        else
            return names
        end
    catch
        return fallback()
    end
end

"Return the number of features `p` in a dataset `X`."
nfeatures(X::AbstractMatrix) = size(X, 2)
nfeatures(X) = length(Tables.columnnames(X))

view_feature(X::AbstractMatrix, feature::Int) = view(X, :, feature)
view_feature(X, feature::Int) = view(X, feature)

"""
Sorting algorithm that ensures that the ordering is always the same.
This means we should avoid `QuickSort` and `PartialQuickSort` as they are not stable.
Only `InsertionSort` and `MergeSort` are stable.
MergeSort has worst case O(N*log N) whereas insertion sort has O(N^2).
"""
const STABLE_SORT_ALG = Base.Sort.MergeSort

end
