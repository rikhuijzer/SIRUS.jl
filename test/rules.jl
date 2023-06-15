let
    text = " X[i, 1] < 1.0 & X[i, 1] ≥ 4.0 "
    @test repr(TreePath(text)) == "TreePath(\"$text\")"
end

let
    text = " X[i, :A] < 1.0 "
    @test_throws ArgumentError repr(TreePath(text))
end

classes = [:a, :b, :c]
left = SIRUS.ClassificationLeaf([1.0, 0.0, 0.0])
feature_name = "1"
splitpoint = SIRUS.SplitPoint(1, Float32(1), feature_name)
right = SIRUS.Node(
            splitpoint,
            SIRUS.ClassificationLeaf([0.0, 1.0, 0.0]),
            SIRUS.ClassificationLeaf([0.0, 0.0, 1.0])
        )

function Rule(path::TreePath, U::AbstractVector, V::AbstractVector)
    T = SIRUS.ClassificationValues
    then_values = T(U)
    else_values = T(V)
    return SIRUS.Rule{T}(path, then_values, else_values)
end

left_rule = Rule(SIRUS.TreePath(" X[i, 1] < 32000 "), [0.61], [0.408])
right_rule = Rule(SIRUS.TreePath(" X[i, 1] ≥ 32000 "), [0.408], [0.61])
@test SIRUS._flip_left([left_rule, right_rule]) == [left_rule, left_rule]

@testset "exported functions" begin
    @test feature_names(left_rule) == ["1"]
    @test directions(left_rule) == [:L]
    @test values(left_rule) == [32000]
end

r1 = Rule(SIRUS.TreePath(" X[i, 1] < 32000 "), [0.61], [0.408])
r1b = Rule(SIRUS.TreePath(" X[i, 1] < 32000 "), [0.61], [0.408])
r1c = Rule(SIRUS.TreePath(" X[i, 1] < 32000 "), [0.0], [0.408])
r5 = Rule(SIRUS.TreePath(" X[i, 3] < 64 "), [0.56], [0.334])

algo = SIRUS.Classification()

let
    expected = [r1 => 2, r5 => 1]
    @test SIRUS._combine_paths(algo, SIRUS._flip_left([r5, r1, r1])) == expected
end
@test SIRUS._mean([[1, 4], [2, 4]]) == [1.5, 4.0]

# @test SIRUS._mode([[1, 2], [1, 6], [4, 6]]) == [1, 6]

splitpoint = SIRUS.SplitPoint(1, Float32(4), feature_name)
node = SIRUS.Node(splitpoint, left, right)

rules = SIRUS._rules!(node)

n = 200
X, y = make_moons(n; rng=_rng(), shuffle=true)
model = StableForestClassifier(; rng=_rng())
mach = machine(model, X, y)
fit!(mach; verbosity=0)
forest = mach.fitresult

rules = SIRUS._rules(forest)

@test hash(r1) == hash(r1b)
@test hash(r1.path) == hash(r1b.path)

@test SIRUS._combine_paths(algo, [r1, r1b]) == [r1 => 2]
expected = SIRUS.ClassificationValues([mean([0.61, 0])])
@test first(only(SIRUS._combine_paths(algo, [r1, r1c]))).then_values == expected
@test SIRUS._count_unique([1, 1, 1, 2]) == Dict(1 => 3, 2 => 1)

weights = [0.395, 0.197, 0.187, 0.057, 0.054, 0.043, 0.027, 0.02, 0.01, 0.01]

empty_model = SIRUS.StableRules(SIRUS.Rule[], algo, [1], Float16[0.1])
@test_throws AssertionError SIRUS._predict(empty_model, [31000])

@test SIRUS._predict(SIRUS.StableRules([r1], algo, [1], Float16[1.0]), [31000]) == [0.61]
@test SIRUS._predict(SIRUS.StableRules([r1], algo, [1], Float16[1.0]), [33000]) == [0.408]
let
    model = SIRUS.StableRules([r1, r5], algo, [1], Float16[0.5, 0.5])
    @test SIRUS._predict(model, [33000, 0, 61]) == [mean([0.408, 0.56])]
end

@test first(SIRUS._process_rules([r5, r1, r1], algo, 10)) == Pair(r1, 2)

@testset "binary show" begin
    r = Rule(SIRUS.TreePath(" X[i, 1] < 5 "), [0.1, 0.9], [0.2, 0.8])
    classes = [0, 1]
    weights = Float16[1.0]
    model = SIRUS.StableRules([r], algo, classes, weights)
    pretty = repr(model)
    @test contains(pretty, "0.9")
    @test contains(pretty, "0.8")
    @test contains(pretty, "showing only")
    @test !contains(pretty, "unexpected")
end

function generate_rules()
    output_type = SIRUS.Classification()
    forest = SIRUS._forest(_rng(), output_type, X, y)
    rulesmodel = let
        rules = SIRUS._rules(forest)
        weights = repeat(Float16[1.0], length(rules))
        SIRUS.StableRules(rules, algo, forest.classes, weights)
    end
    model = StableRulesClassifier(; max_rules=10)
    processed = SIRUS.StableRules(forest, X, y, model)
    (; forest, rulesmodel, processed)
end

generated = map(i -> generate_rules(), 1:10)

"""
Return whether the score for the model is roughly equal to check whether RNG is used correctly.
Checking the scores is easier than the raw models since those seem to differ slightly.
"""
function equal_output(stage::Symbol)
    V = getproperty.(generated, stage)
    P = SIRUS._predict.(V, Ref(X))
    S = auc.(P, Ref(y))
    for i in eachindex(S)
        if !(S[i] ≈ S[1])
            @show i
            return false
        end
    end
    return true
end
@test equal_output(:forest)
@test equal_output(:rulesmodel)
@test equal_output(:processed)

nothing
