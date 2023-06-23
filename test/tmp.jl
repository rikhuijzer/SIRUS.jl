
# From the example in the function docstring.
r1 = S.Rule(S.TreePath(" X[i, 1] < 3 "), [0], [0])
r2 = S.Rule(S.TreePath(" X[i, 1] ≥ 3 "), [0], [0])
r3 = S.Rule(S.TreePath(" X[i, 1] < 3 & X[i, 2] < 2 "), [0], [0])

@test S._tmp_conditions([r1, r2, r3]) == Set([
    S.TreePath(" X[i, 1] < 3 "),
    S.TreePath(" X[i, 1] ≥ 3 "),
    S.TreePath(" X[i, 2] < 2 "),
    S.TreePath(" X[i, 2] ≥ 2 "),
    S.TreePath(" X[i, 1] < 3 & X[i, 2] < 2 "),
])

p1 = S.TreePath(" X[i, 1] < 4 ")

@test S._tmp_implies(r1.path, p1)
@test !S._tmp_implies(p1, r1.path)

p2 = S.TreePath(" X[i, 1] < 3 & X[i, 2] < 2 ")
@test S._tmp_implies(p2, r1.path)
@test !S._tmp_implies(r1.path, p2)

p3 = S.TreePath(" X[i, 2] < 2 ")
@test !S._tmp_implies(p3, r1.path)
@test !S._tmp_implies(r1.path, p3)

@test !S._tmp_implies(r1.path, r2.path)

"Return the index of `v` in `V` and ensure that there is only one match."
function _findonly(f::Function, V)
    indexes = findall(f, V)
    return only(indexes)
end

"Return whether the rules passed into the rule space function imply the condition."
function _condition_implies_rules(condition::S.TreePath, conditions, space::BitMatrix)
    index = _findonly(==(condition), conditions)
    return space[:, index]
end

conditions, space = S._tmp_rule_space([r1, r2, r3])
@test _condition_implies_rules(S.TreePath(" X[i, 1] < 3 "), conditions, space) == Bool[1, 0, 0]
@test _condition_implies_rules(S.TreePath(" X[i, 1] ≥ 3 "), conditions, space) == Bool[0, 1, 0]
@test _condition_implies_rules(S.TreePath(" X[i, 2] < 2 "), conditions, space) == Bool[0, 0, 0]
@test _condition_implies_rules(S.TreePath(" X[i, 2] ≥ 2 "), conditions, space) == Bool[0, 0, 0]
@test _condition_implies_rules(r3.path, conditions, space) == Bool[1, 0, 1]
