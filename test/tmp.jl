
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
@test _condition_implies_rules(S.TreePath(" X[i, 1] < 3 "), conditions, space) == Bool[1, 0, 1]
@test _condition_implies_rules(S.TreePath(" X[i, 1] ≥ 3 "), conditions, space) == Bool[0, 1, 0]
@test _condition_implies_rules(S.TreePath(" X[i, 2] < 2 "), conditions, space) == Bool[0, 0, 1]
@test _condition_implies_rules(S.TreePath(" X[i, 2] ≥ 2 "), conditions, space) == Bool[0, 0, 0]
@test _condition_implies_rules(r3.path, conditions, space) == Bool[0, 0, 1]

###
# TMP COPY FROM TEST/DEPENDENT
###
r1 = S.Rule(S.TreePath(" X[i, 1] < 32000 "), [0.061], [0.408])
r2 = S.Rule(S.TreePath(" X[i, 1] ≥ 32000 "), [0.408], [0.061])

r3 = S.Rule(S.TreePath(" X[i, 2] < 8000 "), [0.062], [0.386])
r4 = S.Rule(S.TreePath(" X[i, 2] ≥ 8000 "), [0.386], [0.062])
r5 = S.Rule(S.TreePath(" X[i, 3] < 64 "), [0.056], [0.334])
r6 = S.Rule(S.TreePath(" X[i, 3] ≥ 64 "), [0.334], [0.056])
r7 = S.Rule(S.TreePath(" X[i, 1] ≥ 32000 & X[i, 3] ≥ 64 "), [0.517], [0.067])
r8 = S.Rule(S.TreePath(" X[i, 4] < 8 "), [0.050], [0.312])
r9 = S.Rule(S.TreePath(" X[i, 4] ≥ 8 "), [0.312], [0.050])
r10 = S.Rule(S.TreePath(" X[i, 5] < 50 "), [0.335], [0.058])
r11 = S.Rule(S.TreePath(" X[i, 5] ≥ 50 "), [0.058], [0.335])
r12 = S.Rule(S.TreePath(" X[i, 1] ≥ 32000 & X[i, 3] < 64 "), [0.192], [0.102])
r13 = S.Rule(S.TreePath(" X[i, 1] < 32000 & X[i, 4] ≥ 8 "), [0.157], [0.100])
# First constraint is updated based on a comment from Clément via email.
r14 = S.Rule(S.TreePath(" X[i, 1] ≥ 32000 & X[i, 4] ≥ 12 "), [0.554], [0.073])
r15 = S.Rule(S.TreePath(" X[i, 1] ≥ 32000 & X[i, 4] < 12 "), [0.192], [0.096])
r16 = S.Rule(S.TreePath(" X[i, 2] ≥ 8000 & X[i, 4] ≥ 12 "), [0.586], [0.76])
r17 = S.Rule(S.TreePath(" X[i, 2] ≥ 8000 & X[i, 4] < 12 "), [0.236], [0.94])

# @test S._tmp_linearly_dependent([r1, r5, r7, r12]) == Bool[0, 0, 0, 1]
# @test S._tmp_filter_linearly_dependent([r1, r5, r7, r12]) == [r1, r5, r7]
# @test S._tmp_filter_linearly_dependent([r1, r5, r12, r7]) == [r1, r5, r7]
