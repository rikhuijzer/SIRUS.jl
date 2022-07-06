_unique_features(split::Split) = split.splitpoint.feature
_unique_features(rule::Rule) = unique(_unique_features.(rule.path.splits))
_unique_features(rules::Vector{Rule}) = unique(reduce(vcat, _unique_features.(rules)))

"""
Return some point which satisifies `rule` and has data for each `unique_features`.
It would also be possible to evaluate whether rules hold without generating points, but it is then tricky to answer constraints consistently.
So, basically, this point generation is a way to encode the information such that the constraints can be consistently answered.
"""
function _point(rule::Rule, unique_features::Vector{Int}, satisfies::Bool)
    point = zeros(length(unique_features))
    prev_f = 0
    for split in _splits(rule)
        threshold = _value(split)
        value = if satisfies
            _direction(split) == :L ? threshold - 1 : threshold
        else
            _direction(split) == :L ? threshold : threshold - 1
        end
        f = _feature(split)
        @assert prev_f != f "The constraints in $rule are on the same feature"
        prev_f = f
        index = findfirst(==(f), unique_features)
        point[index] = value
    end
    return point
end

"""
Return a point which satisifies `A` and `B`.
This assumes that `A` and `B` contain the features in the same order as `_unique_features`.
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
function _feature_space(rules::Vector{Rule}, A::Split, B::Split)
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
function _linearly_redundant(rules::Vector{Rule}, A::Split, B::Split)
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
