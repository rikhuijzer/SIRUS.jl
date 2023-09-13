# These rules are roughly equal to the ones in Table 3 of the Supplementary PDF
# (https://proceedings.mlr.press/v130/benard21a.html).
# Let the features be
# 1: MMAX
# 2: MMIN
# 3: CACH
# 4: CHMIN
# 5: MYCT
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
r16 = S.Rule(S.TreePath(" X[i, 2] ≥ 8000 & X[i, 4] ≥ 12 "), [0.586], [0.076])
r17 = S.Rule(S.TreePath(" X[i, 2] ≥ 8000 & X[i, 4] < 12 "), [0.236], [0.094])

@test S._filter_linearly_dependent([r1, r2, r3, r5]) == [r1, r3, r5]

let
    A = _Split(1, 32000f0, :L)
    B = _Split(3, 64f0, :L)
    rules = [r1, r5, r7, r12]
    expected = Bool[1 1 1 0 0;
                    1 1 0 0 0;
                    1 0 1 0 1;
                    1 0 0 1 0]
    @test S._feature_space(rules, A, B) == expected
    @test S._linearly_dependent(rules, A, B) == Bool[0, 0, 0, 1]
end

let
    A = _Split(1, 32000f0, :L)
    B = _Split(4, 12f0, :L)
    rules = [r1, r14, r15]
    expected = Bool[1 1 0 0;
                    1 1 0 0;
                    1 0 0 1;
                    1 0 1 0]
    @test S._feature_space(rules, A, B) == expected
    @test S._linearly_dependent(rules, A, B) == Bool[0, 0, 1]
end

let
    A = _Split(2, 8000f0, :L)
    B = _Split(4, 12f0, :L)
    rules = [r3, r16, r17]
    expected = Bool[1 1 0 0;
                    1 1 0 0;
                    1 0 0 1;
                    1 0 1 0]
    @test S._feature_space(rules, A, B) == expected
    @test S._linearly_dependent(rules, A, B) == Bool[0, 0, 1]
end

@test S._unique_left_splits([r1, r2]) == [_Split(1, 32000f0, :L)]
let
    expected = [_Split(1, 32000f0, :L), _Split(3, 64f0, :L)]
    @test S._unique_left_splits([r1, r5, r7, r12]) == expected
end

@test S._left_triangular_product([1, 2]) == [(1, 2)]
@test S._left_triangular_product([1, 2, 3]) == [(1, 2), (1, 3), (2, 3)]

let
    A = _Split(2, 8000f0, :L)
    B = _Split(4, 12f0, :L)
    @test S._related_rule(r3, A, B)
    @test S._related_rule(r16, A, B)
    @test S._related_rule(r17, A, B)
    @test !S._related_rule(r1, A, B)
    @test !(S._related_rule(r1, _Split(1, 31000.0f0, :L), B))
end

@test S._filter_linearly_dependent([r1, r3]) == [r1, r3]

@testset "r12 is removed because r7 has a wider gap" begin
    @test Set(S._filter_linearly_dependent([r1, r5, r7, r12])) == Set([r1, r5, r7])
    @test Set(S._filter_linearly_dependent([r1, r5, r12, r7])) == Set([r1, r5, r7])
end

@test Set(S._filter_linearly_dependent([r3, r16, r17])) == Set([r3, r16])
@test Set(S._filter_linearly_dependent([r3, r16, r13])) == Set([r3, r13, r16])

@testset "single rule is not linearly dependent" begin
    A = S.Split(S.SplitPoint(4, 12.0f0, "4"), :L)
    B = S.Split(S.SplitPoint(4, 8.0f0, "4"), :L)
    rule = SIRUS.Rule(TreePath(" X[i, 4] < 8.0 "), [0.05], [0.312])
    @test S._feature_space([rule], A, B)[:, 2] == Bool[1, 0, 1, 0]
    @test S._linearly_dependent([rule], A, B) == Bool[0]
end

@test S._process_rules(repeat([r1], 10), 10) == [r1]

@testset "rank calculation is precise enough" begin
    A = S.Split(S.SplitPoint(2, 8000.0f0, "2"), :L)
    B = S.Split(S.SplitPoint(1, 32000.0f0, "1"), :L)
    n = 34
    dependent = S._linearly_dependent([repeat([r2, r1], 34); r4], A, B)
    expected = Bool[0; repeat([true], 2n-1); 0]
    @test length(dependent) == length(expected)
    @test dependent == expected

    n = 1_000
    dependent = S._linearly_dependent([repeat([r2, r1], n); r4], A, B)
    expected = Bool[0; repeat([true], 2n-1); 0]
    @test length(dependent) == length(expected)
    @test dependent == expected
end

function _canonicalize(rules::Vector{SIRUS.Rule})
    [length(r.path.splits) == 1 ? SIRUS._left_rule(r) : r for r in rules]
end

allrules = [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17]

@testset "_simplify_single_rules is deterministic" begin
    singles = S._simplify_single_rules(allrules)
    for i in 1:1_000
        @test singles == S._simplify_single_rules(allrules)
    end
end

expected = [r1, r3, r5, r7, r8, r10, r13, r14, r16]
actual = S._filter_linearly_dependent(allrules)
@test Set(actual) == Set(expected)

allrules = shuffle(_rng(), allrules)
actual = S._process_rules(allrules, 100)
@test Set(actual) == Set(expected)

@test length(S._filter_linearly_dependent(allrules)) == 9
@test length(S._filter_linearly_dependent(allrules)) == 9
@test length(S._filter_linearly_dependent([r1])) == 1
@test length(S._filter_linearly_dependent(repeat(allrules, 200))) == 9

nothing
