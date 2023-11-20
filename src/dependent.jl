"Return whether `a` implies `b`."
function _implies(a::SubClause, b::SubClause)::Bool
    if _feature(a) == _feature(b)
        if _direction(a) == :L
            if _direction(b) == :L
                return _value(a) ≤ _value(b)
            else
                return false
            end
        else
            if _direction(b) == :R
                return _value(a) ≥ _value(b)
            else
                return false
            end
        end
    else
        return false
    end
end

"""
Return whether `condition` implies `rule`, that is, whether `A & B => rule`.
"""
function _implies(condition::Tuple{SubClause, SubClause}, rule::Rule)
    A, B = condition
    subclauses = _subclauses(rule)
    implied = map(subclauses) do subclause
        _implies(A, subclause) || _implies(B, subclause)
    end
    return all(implied)
end

"""
Return a binary space which can be used to determine whether rules are linearly dependent.

For example, for the conditions

- u: A ≥ 3 & B ≥ 2 (U & V)
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
julia> A = SIRUS.SubClause(SIRUS.SubClause(1, "1", 32000.0f0, :L);

julia> B = SIRUS.SubClause(SIRUS.SubClause(3, "3", 64.0f0, :L);

julia> r1 = SIRUS.Rule(Clause(" X[i, 1] < 32000.0 "), [0.061], [0.408]);

julia> r5 = SIRUS.Rule(Clause(" X[i, 3] < 64.0 "), [0.056], [0.334]);

julia> r7 = SIRUS.Rule(Clause(" X[i, 1] ≥ 32000.0 & X[i, 3] ≥ 64.0 "), [0.517], [0.067]);

julia> r12 = SIRUS.Rule(Clause(" X[i, 1] ≥ 32000.0 & X[i, 3] < 64.0 "), [0.192], [0.102]);

julia> SIRUS.rank(SIRUS._feature_space([r1, r5], A, B))
3

julia> SIRUS.rank(SIRUS._feature_space([r1, r5, r7], A, B))
4

julia> SIRUS.rank(SIRUS._feature_space([r1, r5, r7, r12], A, B))
4
```
"""
function _feature_space(rules::AbstractVector{Rule}, A::SubClause, B::SubClause)::BitMatrix
    l = length(rules)
    data = BitMatrix(undef, 4, l + 1)
    for i in 1:4
        data[i, 1] = 1
    end

    nA = _reverse(A)
    nB = _reverse(B)
    for col in 2:l+1
        rule = rules[col-1]
        data[1, col] = _implies((A, B), rule)
        data[2, col] = _implies((A, nB), rule)
        data[3, col] = _implies((nA, B), rule)
        data[4, col] = _implies((nA, nB), rule)
    end
    return data
end

"Canonicalize a SubClause by ensuring that the direction is left."
_canonicalize(s::SubClause) = _direction(s) == :L ? s : _reverse(s)

"""
Return a vector of unique left splits for `rules`.
These splits will be used to form `(A, B)` pairs and generate the feature space.
For example, the pair `x[i, 1] < 32000` (A) and `x[i, 3] < 64` (B) will be used to generate
the feature space `A & B`, `A & !B`, `!A & B`, `!A & !B`.
"""
function _unique_left_subclauses(rules::Vector{Rule})
    subclauses = SubClause[]
    for rule in rules
        for subclause in _subclauses(rule)
            canonicalized = _canonicalize(subclause)
            if !(canonicalized in subclauses)
                push!(subclauses, canonicalized)
            end
        end
    end
    return subclauses
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
            if i < j
                right = V[j]
                push!(product, (left, right))
            end
        end
    end
    return product
end

"""
Return whether some rule is either related to `A` or `B` or both.
Here, it is very important to get rid of rules which are about the same feature but different thresholds.
Otherwise, rules will be wrongly classified as linearly dependent in the next step.
"""
function _related_rule(rule::Rule, A::SubClause, B::SubClause)::Bool
    @assert _direction(A) == :L
    @assert _direction(B) == :L
    subclauses = _subclauses(rule)
    if length(subclauses) == 1
        subclause = only(subclauses)
        left_subclause = _canonicalize(subclause)
        return left_subclause == A || left_subclause == B
    elseif length(subclauses) == 2
        l1 = _canonicalize(subclauses[1])
        l2 = _canonicalize(subclauses[2])
        return (l1 == A && l2 == B) || (l1 == B && l2 == A)
    else
        @error "Rule $rule has more than two splits; this is not supported."
    end
end

"""
Return a vector of booleans with a true for every rule in `rules` that is linearly dependent on a combination of the previous rules.
To find rules for this method, collect all rules containing some feature for each pair of features.
That should be a fairly quick way to find subsets that are easy to process.
"""
function _linearly_dependent(
        rules::AbstractVector{Rule},
        A::SubClause,
        B::SubClause
    )::BitArray
    data = _feature_space(rules, A, B)
    l = length(rules)
    dependent = BitArray(undef, l)
    atol = 1e-6
    current_rank = rank(data[:, 1:1]; atol)
    for i in 1:l
        new_rank = rank(view(data, :, 1:i+1); atol)
        if current_rank < new_rank
            dependent[i] = false
            current_rank = new_rank
        else
            dependent[i] = true
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
    alg = Helpers.STABLE_SORT_ALG
    return sort(rules; alg, by=_gap_size, rev=true)
end

"""
Simplify the rules that contain a single split by only retaining rules that point left and
removing duplicates.
"""
function _simplify_single_rules(rules::Vector{Rule})::Vector{Rule}
    out = OrderedSet{Rule}()
    for rule in rules
        splits = _subclauses(rule)
        if length(splits) == 1
            left_rule = _left_rule(rule)
            push!(out, left_rule)
        else
            push!(out, rule)
        end
    end
    return collect(out)
end

"""
Return a vector of rules that are not linearly dependent on any other rule.

This is done by considering each pair of splits.
For example, considers the pair `x[i, 1] < 32000` (A) and `x[i, 3] < 64` (B).
Then, for each rule, it checks whether the rule is linearly dependent on the pair.
As soon as a dependent rule is found, it is removed from the set to avoid considering it again.
If we don't do this, we might remove some rule `r` that causes another rule to be linearly
dependent in one related set, but then is removed in another related set.
"""
function _filter_linearly_dependent(rules::Vector{Rule})::Vector{Rule}
    sorted = _sort_by_gap_size(rules)
    S = _unique_left_subclauses(sorted)
    pairs = _left_triangular_product(S)
    out = copy(sorted)
    for (A, B) in pairs
        indexes = filter(i -> _related_rule(out[i], A, B), 1:length(out))
        subset = view(out, indexes)
        dependent_subset = _linearly_dependent(subset, A, B)
        @assert length(indexes) == length(subset)
        @assert length(dependent_subset) == length(subset)
        dependent_indexes = indexes[dependent_subset]
        alg = Helpers.STABLE_SORT_ALG
        deleteat!(out, sort(dependent_indexes; alg))
    end
    return out
end
