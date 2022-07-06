# These rules are roughly equal to the ones in Table 3 of the Supplementary PDF
# (https://proceedings.mlr.press/v130/benard21a.html).
# Let the features be
# 1: MMAX
# 2: MMIN
# 3: CACH
# 4: CHMIN
# 5: MYCT
r1 = ST.Rule(ST.TreePath(" X[i, 1] < 32000 "), [0.61], [0.408])
r2 = ST.Rule(ST.TreePath(" X[i, 1] ≥ 32000 "), [0.408], [0.61])

@test ST._filter_reversed([r1, r2]) == [r1]

r3 = ST.Rule(ST.TreePath(" X[i, 2] < 8000 "), [0.62], [0.386])
r5 = ST.Rule(ST.TreePath(" X[i, 3] < 64 "), [0.56], [0.334])
r7 = ST.Rule(ST.TreePath(" X[i, 1] ≥ 32000 & X[i, 3] ≥ 64 "), [0.517], [0.67])
r8 = ST.Rule(ST.TreePath(" X[i, 4] < 8 "), [0.50], [0.312])
r10 = ST.Rule(ST.TreePath(" X[i, 5] < 50 "), [0.335], [0.58])
r12 = ST.Rule(ST.TreePath(" X[i, 1] ≥ 32000 & X[i, 3] < 64 "), [0.192], [0.102])
r13 = ST.Rule(ST.TreePath(" X[i, 1] < 32000 & X[i, 4] ≥ 12 "), [0.554], [0.73])
# First constraint is updated based on a comment from Clément via email.
r14 = ST.Rule(ST.TreePath(" X[i, 1] ≥ 32000 & X[i, 4] ≥ 12 "), [0.192], [0.102])
r15 = ST.Rule(ST.TreePath(" X[i, 1] ≥ 32000 & X[i, 4] < 12 "), [0.192], [0.102])
r16 = ST.Rule(ST.TreePath(" X[i, 2] ≥ 8000 & X[i, 4] ≥ 12 "), [0.586], [0.76])
r17 = ST.Rule(ST.TreePath(" X[i, 2] ≥ 8000 & X[i, 4] < 12 "), [0.236], [0.94])

@test ST._unique_features([r1, r7, r12]) == [1, 3]
@test sort(ST._unique_features([r1, r7, r12, r17])) == [1, 2, 3, 4]

@test ST._point(r1, [1], true) == [31999.0]
@test ST._point(r1, [1, 2], true) == [31999.0, 0.0]
@test ST._point(r7, [1, 3], true) == [32000.0, 64.0]
@test ST._point(r17, [2, 4], true) == [8000.0, 11.0]
@test ST._point(r1, [1], false) == [32000.0]
@test ST._point(r17, [2, 4], false) == [7999.0, 12.0]

@test ST._satisfies([1], [31999.0], r1)
@test !ST._satisfies([1], [32000.0], r1)
@test ST._satisfies([1, 3], [32000.0, 64.0], r7)
@test !ST._satisfies([1, 3], [31999.0, 64.0], r7)
@test !ST._satisfies([1, 3], [31999.0, 63.0], r7)

let
    A = ST.Split(1, 32000f0, :L)
    B = ST.Split(3, 64f0, :L)
    rules = [r1, r5, r7, r12]
    expected = Bool[1 1 1 0 0;
                    1 1 0 0 0;
                    1 0 1 0 1;
                    1 0 0 1 0]
    @test ST._feature_space(rules, A, B) == expected
    @test ST._linearly_redundant(rules, A, B) == Bool[0, 0, 0, 1]
end

let
    A = ST.Split(1, 32000f0, :L)
    B = ST.Split(4, 12f0, :L)
    rules = [r1, r14, r15]
    expected = Bool[1 1 0 0;
                    1 1 0 0;
                    1 0 0 1;
                    1 0 1 0]
    @test ST._feature_space(rules, A, B) == expected
    @test ST._linearly_redundant(rules, A, B) == Bool[0, 0, 1]
end

let
    A = ST.Split(2, 8000f0, :L)
    B = ST.Split(4, 12f0, :L)
    rules = [r3, r16, r17]
    expected = Bool[1 1 0 0;
                    1 1 0 0;
                    1 0 0 1;
                    1 0 1 0]
    @test ST._feature_space(rules, A, B) == expected
    @test ST._linearly_redundant(rules, A, B) == Bool[0, 0, 1]
end

@test ST._equal_variables_thresholds(r1, r2) == true
@test ST._equal_variables_thresholds(r1, r5) == false
@test ST._equal_variables_thresholds(r5, r12) == false
@test ST._equal_variables_thresholds(r7, r12) == true
@test ST._equal_variables_thresholds(r12, r15) == false

@test ST._gap_width(r12) < ST._gap_width(r7)
ST._linearly_redundant([r1, r3]) == Bool[0, 0]
@test ST._linearly_redundant([r1, r5, r7, r12]) == Bool[0, 0, 0, 1]

# @test ST._linearly_redundant([r3, r16, r17]) == Bool[0, 0, 1]

