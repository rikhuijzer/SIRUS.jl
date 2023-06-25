_unique_features(split::Split) = split.splitpoint.feature
_unique_features(rule::Rule) = unique(_unique_features.(rule.path.splits))
_unique_features(rules::Vector{Rule}) = unique(reduce(vcat, _unique_features.(rules)))

"""
Return a point which satisifies `A` and `B`.
This assumes that `A` and `B` contain the features in the same order as `_unique_features`.
Basically, this point generation is a way to encode the information such that the constraints can be consistently answered.
"""
function _point(A::Split, B::Split)
    va = _value(A)
    a = _direction(A) == :L ? va - 1 : va
    vb = _value(B)
    b = _direction(B) == :L ? vb - 1 : vb
    return [a, b]
end

"Return whether `point` satisifies `rule`."
function _satisfies(unique_features::Vector{Int}, point::Vector, rule::Rule)
    for split in _splits(rule)
        index = findfirst(==(_feature(split)), unique_features)
        value = point[index]
        threshold = _value(split)
        if !(_direction(split) == :L ? value < threshold : value ≥ threshold)
            return false
        end
    end
    return true
end

function _reduced_echelon_form(A::AbstractMatrix)
    rref(A)
end

"""
Return a binary space which can be used to determine whether rules are linearly dependent.

For example, for the conditions

- u:  ≥ 3 & B ≥ 2 (U & V)
- v: A ≥ 3 & B < 2 (U & !V)
- w: A < 3 & B ≥ 2 (!U & V)
- x: A < 3 & B < 2 (!U & !V)

For example, given the following rules:

Rule 1: x[i, 1] < 32000
Rule 5: x[i, 3] < 64
Rule 7: x[i, 1] ≥ 32000 & x[i, 3] < 64
Rule 12: x[i, 1] < 32000 & x[i, 3] < 64

and the following clauses

A: x[i, 1] < 32000
B: x[i, 3] < 64

This function generates a matrix containing a row for

- A && B (x[i, 1] < 32000 & x[i, 3] < 64)
- A && !B (x[i, 1] < 32000 & x[i, 3] ≥ 64)
- !A && B (x[i, 1] ≥ 32000 & x[i, 3] < 64)
- !A && !B (x[i, 1] ≥ 32000 & x[i, 3] ≥ 64)

and one zeroes column:

| Condition  | Ones | R1 | R5 | R7 | R12 |
| ---------- | ---- | -- | -- | -- | --- |
|   A &&  B  |   1  |  1 |  1 |  0 |  0  |
|   A && !B  |   1  |  1 |  0 |  0 |  0  |
|  !A &&  B  |   1  |  0 |  1 |  0 |  1  |
|  !A && !B  |   1  |  0 |  0 |  1 |  0  |

In other words, the matrix represents which rules are implied by each syntetic datapoint
(conditions in the rows).
Next, this can be used to determine which rules are linearly dependent by checking whether
the rank increases when adding rules.

# Example

```jldoctest
julia> A = SIRUS.Split(SIRUS.SplitPoint(1, 32000.0f0, "1"), :L);

julia> B = SIRUS.Split(SIRUS.SplitPoint(3, 64.0f0, "3"), :L);

julia> r1 = SIRUS.Rule(TreePath(" X[i, 1] < 32000.0 "), [0.061], [0.408]);

julia> r5 = SIRUS.Rule(TreePath(" X[i, 3] < 64.0 "), [0.056], [0.334]);

julia> r7 = SIRUS.Rule(TreePath(" X[i, 1] ≥ 32000.0 & X[i, 3] ≥ 64.0 "), [0.517], [0.067]);

julia> r12 = SIRUS.Rule(TreePath(" X[i, 1] ≥ 32000.0 & X[i, 3] < 64.0 "), [0.192], [0.102]);

julia> SIRUS.rank(SIRUS._feature_space([r1, r5], A, B))
3

julia> SIRUS.rank(SIRUS._feature_space([r1, r5, r7], A, B))
4

julia> SIRUS.rank(SIRUS._feature_space([r1, r5, r7, r12], A, B))
4
```
"""
function _feature_space(rules::AbstractVector{Rule}, A::Split, B::Split)::BitMatrix
    l = length(rules)
    data = BitMatrix(undef, 4, l + 1)
    for i in 1:4
        data[i, 1] = 1
    end

    F = [_feature(A), _feature(B)]
    nA = _reverse(A)
    nB = _reverse(B)
    for col in 2:l+1
        rule = rules[col-1]
        data[1, col] = _satisfies(F, _point(A, B), rule)
        data[2, col] = _satisfies(F, _point(A, nB), rule)
        data[3, col] = _satisfies(F, _point(nA, B), rule)
        data[4, col] = _satisfies(F, _point(nA, nB), rule)
    end
    return data
end

# Temporary function to work for finding linearly dependent rules
function _tmpldep(rules::AbstractVector{Rule})
    data = _feature_space(rules, A, B)
    l = length(rules)
    
end

"""
Return a vector of unique left splits for `rules`.
These splits are required to form `[A, B]` pairs in the next step.
"""
function _unique_left_splits(rules::Vector{Rule})
    splits = Split[]
    for rule in rules
        for split in _splits(rule)
            left_split = _direction(split) == :L ? split : _reverse(split)
            if !(left_split in splits)
                push!(splits, left_split)
            end
        end
    end
    return splits
end

"""
Return all unique pairs of elements in `V`.
More formally, return all pairs (v_i, v_j) where i < j.
"""
function _left_triangular_product(V::Vector{T}) where {T}
    l = length(V)
    product = Tuple{T,T}[]
    for i in 1:l
        left = V[i]
        for j in 1:l
            right = V[j]
            if i < j
                push!(product, (left, right))
            end
        end
    end
    return product
end

_left_split(s::Split) = _direction(s) == :L ? s : _reverse(s)

"""
Return whether some rule is either related to `A` or `B` or both.
Here, it is very important to get rid of rules which are about the same feature but different thresholds.
Otherwise, rules will be wrongly classified as linearly dependent in the next step.
"""
function _related_rule(rule::Rule, A::Split, B::Split)::Bool
    @assert _direction(A) == :L
    @assert _direction(B) == :L
    splits = _splits(rule)
    fa = _feature(A)
    fb = _feature(B)
    if length(splits) == 1
        split = only(splits)
        left_split = _left_split(split)
        return left_split == A || left_split == B
    else
        l1 = _left_split(splits[1])
        l2 = _left_split(splits[2])
        return (l1 == A && l2 == B) || (l1 == B && l2 == A)
    end
end

# function _related_rules(rule::Rule, 

"""
Return a vector of booleans with a true for every rule in `rules` that is linearly dependent on a combination of the previous rules.
To find rules for this method, collect all rules containing some feature for each pair of features.
That should be a fairly quick way to find subsets that are easy to process.
"""
function _linearly_dependent(
        rules::AbstractVector{Rule},
        A::Split,
        B::Split
    )::BitArray
    data = _feature_space(rules, A, B)
    l = length(rules)
    dependent = BitArray(undef, l)
    result = 1
    for i in 1:l
        new_result = rank(view(data, :, 1:i+1))
        rank_increased = new_result == result + 1
        if rank_increased
            result = new_result
            dependent[i] = false
        else
            result = new_result
            dependent[i] = true
        end
    end
    return dependent
end

function _linearly_dependent(rules::Vector{Rule})::BitVector
    S = _unique_left_splits(rules)
    P = _left_triangular_product(S)
    # A `BitVector(undef, length(rules))` here will cause randomness.
    dependent = falses(length(rules))
    for (A, B) in P
        indexes = filter(i -> _related_rule(rules[i], A, B), 1:length(rules))
        subset = view(rules, indexes)
        dependent_subset = _linearly_dependent(subset, A, B)
        # Then note which rule can be removed and filter those in the next step.
        for i in 1:length(dependent_subset)
            if dependent_subset[i]
                dependent[indexes[i]] = true
            end
        end
    end
    return dependent
end

function _gap_size(rule::Rule)
    @assert length(rule.then) == length(rule.otherwise)
    gap_size_per_class = abs.(rule.then .- rule.otherwise)
    sum(gap_size_per_class)
end

"""
Return the vector rule sorted by decreasing gap size.
This allows the linearly dependent filter to remove the rules further down the list since
they have a smaller gap.
"""
function _sort_by_gap_size(rules::Vector{Rule})
    return sort(rules; by=_gap_size, rev=true)
end

"""
Return the subset of `rules` which are not linearly dependent.
This is based on a complex heuristic involving calculating the rank of the matrix, see above StackExchange link for more information.
"""
function _filter_linearly_dependent(rules::Vector{Rule})::Vector{Rule}
    sorted = _sort_by_gap_size(rules)
    dependent = _linearly_dependent(sorted)
    out = Rule[]
    for i in 1:length(dependent)
        if !dependent[i]
            push!(out, rules[i])
        end
    end
    return out
end
