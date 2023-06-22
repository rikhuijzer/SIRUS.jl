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

"""
"""
function _feature_space(rules::AbstractVector{Rule}, A::Split, B::Split)
    l = length(rules)
    data = BitArray(undef, 4, l + 1)
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

"Return the product of `V` and `V` for all pairs (v_i, v_j) where i < j."
function _left_triangular_product(V::Vector{T}) where {T}
    l = length(V)
    nl = l - 1
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

Assumes that both `_direction(A)` and `_direction(B)` are `:L`.
"""
function _related_rule(rule::Rule, A::Split, B::Split)
    splits = _splits(rule)
    fa = _feature(A)
    fb = _feature(B)
    if length(splits) == 1
        split = splits[1]
        left_split = _left_split(split)
        return left_split == A || left_split == B
    else
        l1 = _left_split(splits[1])
        l2 = _left_split(splits[2])
        return (l1 == A && l2 == B) || (l1 == B && l2 == A)
    end
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
        # Only allow setting true to avoid setting things to false.
        for i in 1:length(dependent_subset)
            if dependent_subset[i]
                dependent[indexes[i]] = true
            end
        end
    end
    return dependent
end

"""
Return the subset of `rules` which are not linearly dependent.
This is based on a complex heuristic involving calculating the rank of the matrix, see above StackExchange link for more information.
Also note that this method assumes that the rules are assumed to be in ordered by frequency of occurence in the trees.
This assumption is used to filter less common rules when finding linearly dependent rules.
"""
function _filter_linearly_dependent(rules::Vector{Rule})::Vector{Rule}
    dependent = _linearly_dependent(rules)
    out = Rule[]
    for i in 1:length(dependent)
        if !dependent[i]
            push!(out, rules[i])
        end
    end
    return out
end
