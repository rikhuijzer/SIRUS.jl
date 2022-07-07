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
Return whether each rule in `rules` is linearly dependent on a combination of rules before it.
This works by iteratively calculating the rank and seeing whether the rank increases.

To generate points for the rank calculation, assume that we are limited to a set of rules where either `A & B`, `A & !B`, `!A & B`, `!A & !B`, `A`, `!A`, `B`, `!B` or `True`.
This last case is not a valid rule in this algorithm, so that will not happen.
Now, given `A` and `B`, we can create a binary matrix with a row for `A & B`, `A & !B`, `!A & B`, `!A & !B`.
Next, generate one column containing `true`s and one column for each rule in `rules`.
In each column, answer whether the rule holds for some point that satisifies the conditional.
This trick of taking specifically these rows is briliant.
Credits go to D.W. on StackExchange (https://cs.stackexchange.com/a/152819/98402).
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
function _linearly_dependent(rules::AbstractVector{Rule}, A::Split, B::Split)
    data = _feature_space(rules, A, B)
    l = length(rules)
    results = BitArray(undef, l)
    result = 1
    for i in 1:l
        new_result = rank(view(data, :, 1:i+1))
        if new_result == result + 1
            result = new_result
            results[i] = 0
        else
            result = new_result
            results[i] = 1
        end
    end
    return results
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

function _linearly_dependent(rules::Vector{Rule})
    S = _unique_left_splits(rules)
    P = _left_triangular_product(S)
    dependent = BitVector(undef, length(rules))
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
Note that calculating the rank is expensive, so make sure to pre-filter before calling this function.
Also note that this method assumes that the rules are assumed to be in ordered by frequency of occurence in the trees.
This assumption is used to filter less common rules when finding linearly dependent rules.
"""
function _filter_linearly_dependent_rank(rules::Vector{Rule})
    dependent = _linearly_dependent(rules)
    out = Rule[]
    for i in 1:length(dependent)
        if !dependent[i]
            push!(out, rules[i])
        end
    end
    return out
end

"""
Return a subset of `rules` where simple duplicates have been removed.
Also, this flips all rules containing a single clause to the left.
This heuristic is more quick than finding linearly dependent rules.
"""
function _prefilter_linearly_dependent(rules::Vector{Rule})
    out = Rule[]
    for rule in rules
        splits = _splits(rule)
        if length(splits) == 1
            left_rule = _left_rule(rule)
            if !(left_rule in out)
                push!(out, left_rule)
            end
        else
            if !(rule in out)
                push!(out, rule)
            end
        end
    end
    return out
end

function _filter_linearly_dependent(rules::Vector{Rule})
    prefiltered = _prefilter_linearly_dependent(rules)
    return _filter_linearly_dependent_rank(prefiltered)
end
