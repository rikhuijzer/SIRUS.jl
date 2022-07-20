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

r3 = ST.Rule(ST.TreePath(" X[i, 2] < 8000 "), [0.62], [0.386])
r4 = ST.Rule(ST.TreePath(" X[i, 2] ≥ 8000 "), [0.386], [0.62])
r5 = ST.Rule(ST.TreePath(" X[i, 3] < 64 "), [0.56], [0.334])
r6 = ST.Rule(ST.TreePath(" X[i, 3] ≥ 64 "), [0.334], [0.56])
r7 = ST.Rule(ST.TreePath(" X[i, 1] ≥ 32000 & X[i, 3] ≥ 64 "), [0.517], [0.67])
r8 = ST.Rule(ST.TreePath(" X[i, 4] < 8 "), [0.50], [0.312])
r9 = ST.Rule(ST.TreePath(" X[i, 4] ≥ 8 "), [0.312], [0.50])
r10 = ST.Rule(ST.TreePath(" X[i, 5] < 50 "), [0.335], [0.58])
r11 = ST.Rule(ST.TreePath(" X[i, 5] ≥ 50 "), [0.58], [0.335])
r12 = ST.Rule(ST.TreePath(" X[i, 1] ≥ 32000 & X[i, 3] < 64 "), [0.192], [0.102])
r13 = ST.Rule(ST.TreePath(" X[i, 1] < 32000 & X[i, 4] ≥ 8 "), [0.554], [0.73])
# First constraint is updated based on a comment from Clément via email.
r14 = ST.Rule(ST.TreePath(" X[i, 1] ≥ 32000 & X[i, 4] ≥ 12 "), [0.192], [0.102])
r15 = ST.Rule(ST.TreePath(" X[i, 1] ≥ 32000 & X[i, 4] < 12 "), [0.192], [0.102])
r16 = ST.Rule(ST.TreePath(" X[i, 2] ≥ 8000 & X[i, 4] ≥ 12 "), [0.586], [0.76])
r17 = ST.Rule(ST.TreePath(" X[i, 2] ≥ 8000 & X[i, 4] < 12 "), [0.236], [0.94])

@test ST._unique_features([r1, r7, r12]) == [1, 3]
@test sort(ST._unique_features([r1, r7, r12, r17])) == [1, 2, 3, 4]

"Add the frequencies and remove them again after filtering."
function wrap_filter(rules::Vector{ST.Rule})
    pairs = [rule => 0 for rule in rules]
    filtered = ST._filter_linearly_dependent(pairs)
    return first.(filtered)
end

@test wrap_filter([r1, r2, r3, r5]) == [r1, r3, r5]

let
    A = _Split(1, 32000f0, :L)
    B = _Split(3, 64f0, :L)
    rules = [r1, r5, r7, r12]
    expected = Bool[1 1 1 0 0;
                    1 1 0 0 0;
                    1 0 1 0 1;
                    1 0 0 1 0]
    @test ST._feature_space(rules, A, B) == expected
    @test ST._linearly_dependent(rules, A, B) == Bool[0, 0, 0, 1]
end

let
    A = _Split(1, 32000f0, :L)
    B = _Split(4, 12f0, :L)
    rules = [r1, r14, r15]
    expected = Bool[1 1 0 0;
                    1 1 0 0;
                    1 0 0 1;
                    1 0 1 0]
    @test ST._feature_space(rules, A, B) == expected
    @test ST._linearly_dependent(rules, A, B) == Bool[0, 0, 1]
end

let
    A = _Split(2, 8000f0, :L)
    B = _Split(4, 12f0, :L)
    rules = [r3, r16, r17]
    expected = Bool[1 1 0 0;
                    1 1 0 0;
                    1 0 0 1;
                    1 0 1 0]
    @test ST._feature_space(rules, A, B) == expected
    @test ST._linearly_dependent(rules, A, B) == Bool[0, 0, 1]
end

@test ST._unique_left_splits([r1, r2]) == [_Split(1, 32000f0, :L)]
let
    expected = [_Split(1, 32000f0, :L), _Split(3, 64f0, :L)]
    @test ST._unique_left_splits([r1, r5, r7, r12]) == expected
end

@test ST._left_triangular_product([1, 2]) == [(1, 2)]
@test ST._left_triangular_product([1, 2, 3]) == [(1, 2), (1, 3), (2, 3)]

let
    A = _Split(2, 8000f0, :L)
    B = _Split(4, 12f0, :L)
    @test ST._related_rule(r3, A, B)
    @test ST._related_rule(r16, A, B)
    @test ST._related_rule(r17, A, B)
    @test !ST._related_rule(r1, A, B)
    @test !(ST._related_rule(r1, _Split(1, 31000.0f0, :L), B))
end

@test ST._linearly_dependent([r1, r3]) == Bool[0, 0]
@test ST._linearly_dependent([r1, r5, r7, r12]) == Bool[0, 0, 0, 1]

@test wrap_filter([r1, r5, r7, r12]) == [r1, r5, r7]

@test ST._linearly_dependent([r3, r16, r17]) == Bool[0, 0, 1]
@test ST._linearly_dependent([r3, r16, r13]) == Bool[0, 0, 0]

let
    allrules = [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17]
    expected = [r1, r3, r5, r7, r8, r10, r13, r14, r16]
    @test wrap_filter(allrules) == expected

    @test length(ST._process_rules(allrules, 9)) == 9
    @test length(ST._process_rules(allrules, 10)) == 9
    @test length(ST._process_rules([r1], 9)) == 1
    @test length(ST._process_rules(repeat(allrules, 200), 9)) == 9
end

nothing
