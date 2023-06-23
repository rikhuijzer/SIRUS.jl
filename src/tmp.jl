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
    double_conditions = _tmp_double_conditions(rules)
    return union(single_conditions, double_conditions)
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

"""
Return a binary matrix containing the rules (rows) and the clauses from the rules (columns).

This is done by adding a column for each separate clause including `A` if the set contains `A & B`.
And also adding a column for each conjunction (&).

For example, for the rule set

Rule 1: A < 3, then ...
Rule 2: A ≥ 3, then ...
Rule 3: A < 3 & B < 2, then ...

returns the following matrix (without the headers):

| ---- | A < 3 | A ≥ 3 | B < 2 | B ≥ 2 | A < 3 & B < 2 |
| ---- | ----- | ----- | ----- | ----- | ------------- |
|  R1  |   1   |   0   |   0   |   0   |       1       |
|  R2  |   0   |   1   |   0   |   0   |       0       |
|  R3  |   0   |   0   |   0   |   0   |       1       |

Note that the unknown cases (A < 3 => B < 2?) are set to 0.
Gaussian elimination needs to know only implications (=>).
"""
function _tmp_rule_space(rules::Vector{Rule})
    conditions = collect(_tmp_conditions(rules))::Vector{TreePath}
    space = falses(length(rules), length(conditions))
    for i in eachindex(rules)
        rule = rules[i]
        for j in eachindex(conditions)
            condition = conditions[j]
            space[i, j] = _tmp_implies(condition, rule.path)
        end
    end
    return (conditions, space)
end

function _tmp_linearly_dependent(rules::Vector{Rule})
    
end
