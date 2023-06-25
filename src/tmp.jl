function _tmp_single_conditions(rules::Vector{Rule})
    conditions = Set{TreePath}()
    for rule in rules
        for split in _splits(rule)
            push!(conditions, TreePath([split]))
            reversed = _reverse(split)
            push!(conditions, TreePath([reversed]))
        end
    end
    return conditions
end

function _tmp_double_conditions(rules::Vector{Rule})
    conditions = Set{TreePath}()
    for rule in rules
        if 1 < length(_splits(rule))
            push!(conditions, rule.path)
        end
    end
    return conditions
end

"""
Return all the conditions from the rules to be used in the rule space.
Each separate condition from the set of `rules` is returned including `A` if the set contains `A & B`.

For example, for the rule set

Rule 1: A < 3, then ...
Rule 2: A ≥ 3, then ...
Rule 3: A < 3 & B < 2, then ...

Return the following conditions:

- A < 3
- A ≥ 3
- B < 2
- B ≥ 2
- A < 3 & B < 2
"""
function _tmp_conditions(rules::Vector{Rule})
    single_conditions = _tmp_single_conditions(rules)
    # double_conditions = _tmp_double_conditions(rules)
    # return union(single_conditions, double_conditions)
end

"Return whether `clause1` implies `clause2`."
function _tmp_implies(clause1::Split, clause2::Split)::Bool
    if _feature(clause1) == _feature(clause2)
        if _direction(clause1) == :L
            if _direction(clause2) == :L
                return _value(clause1) ≤ _value(clause2)
            else
                return false
            end
        else
            if _direction(clause2) == :R
                return _value(clause1) ≥ _value(clause2)
            else
                return false
            end
        end
    else
        return false
    end
end

"Return whether `condition` implies `clause`."
function _tmp_implies(condition::TreePath, clause::Split)::Bool
    covered = (_tmp_implies(c, clause) for c in _splits(condition))
    return any(covered)
end

"""
Return whether `condition1` implies `condition2`.
Here, the word _implication_ for "A => B" is used in the formal logical meaning
as in "if A is true then B must also be true".

# Example

```julia
julia> a = S.TreePath(" X[i, 1] < 3 ");

julia> b = S.TreePath(" X[i, 1] < 4 ");

julia> S._tmp_implies(a, b)
true
```
"""
function _tmp_implies(condition1::TreePath, condition2::TreePath)::Bool
    # For `condition1` to imply `condition2`, each clause in `condition2` must be implied
    # by a clause in `condition1`.
    covered = map(_splits(condition2)) do clause
        any(_tmp_implies(condition1, clause))
    end
    return all(covered)
end

function _tmp_rule_space(rules::Vector{Rule})
    conditions = collect(_tmp_conditions(rules))::Vector{TreePath}
    space = falses(length(rules), length(conditions))
    for i in eachindex(rules)
        rule = rules[i]
        for j in eachindex(conditions)
            condition = conditions[j]
            space[i, j] = _tmp_implies(rule.path, condition)
        end
    end
    return (conditions, space)
end

"Return the indexes of the linearly dependent rules."
function _tmp_linearly_dependent(rules::Vector{Rule})
    @assert _tmp_gap_size(rules[end]) ≤ _tmp_gap_size(rules[1])
    conditions, space = _tmp_rule_space(rules)
    n_rules = size(space, 1)
    n_conditions = size(space, 2)
    @assert n_conditions == length(conditions)
    reduced_form = _reduced_echelon_form(space)
    findall(x -> all(iszero, x), eachrow(reduced_form))
end

function _tmp_gap_size(rule::Rule)
    @assert length(rule.then) == length(rule.otherwise)
    gap_size_per_class = abs.(rule.then .- rule.otherwise)
    sum(gap_size_per_class)
end

"""
Return the vector rule sorted by decreasing gap size.
This allows the linearly dependent filter to remove the rules further down the list since
they have a smaller gap.
"""
function _tmp_sort_by_gap_size(rules::Vector{Rule})
    return sort(rules; by=_tmp_gap_size, rev=true)
end

"""
Return `rules` but with linearly dependent rules removed.
Note that this does not remove the rules with one constraint which are identical to a
previous rule with the constraint sign reversed.
"""
function _tmp_filter_linearly_dependent(rules::Vector{Rule})::Vector{Rule}
    sorted = _tmp_sort_by_gap_size(rules)
    indexes = _tmp_linearly_dependent(sorted)
    return sorted[setdiff(1:length(sorted), indexes)]
end
