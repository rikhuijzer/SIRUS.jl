let
    text = " X[i, 1] < 1.0 & X[i, 1] ≥ 4.0 "
    @test repr(S.Clause(text)) == "Clause(\"$text\")"
end

let
    text = " X[i, :A] < 1.0 "
    @test_throws ArgumentError repr(S.Clause(text))
end

classes = [:a, :b, :c]
left = S.ClassificationLeaf([1.0, 0.0, 0.0])
feature_name = "1"
splitpoint = S.SplitPoint(1, Float32(1), feature_name)
right = S.Node(
            splitpoint,
            S.ClassificationLeaf([0.0, 1.0, 0.0]),
            S.ClassificationLeaf([0.0, 0.0, 1.0])
        )

left_rule = S.Rule(S.Clause(" X[i, 1] < 32000 "), [0.61], [0.408])

@testset "exported functions" begin
    @test feature_names(left_rule) == ["1"]
    @test directions(left_rule) == [:L]
    @test values(left_rule) == [32000]
end

r1 = S.Rule(S.Clause(" X[i, 1] < 32000 "), [0.61], [0.408])
r1b = S.Rule(S.Clause(" X[i, 1] < 32000 "), [0.61], [0.408])
r1c = S.Rule(S.Clause(" X[i, 1] < 32000 "), [0.0], [0.408])
r5 = S.Rule(S.Clause(" X[i, 3] < 64 "), [0.56], [0.334])

algo = SIRUS.Classification()
@test S._mean([[1, 4], [2, 4]]) == [1.5, 4.0]

# @test S._mode([[1, 2], [1, 6], [4, 6]]) == [1, 6]

splitpoint = S.SplitPoint(1, Float32(4), feature_name)
node = S.Node(splitpoint, left, right)

rules = S._rules!(node)

n = 200
X, y = make_moons(n; rng=_rng(), shuffle=true)
model = StableForestClassifier(; rng=_rng())
mach = machine(model, X, y)
fit!(mach; verbosity=0)
forest = mach.fitresult

rules = S._rules(forest)

@test hash(r1) == hash(r1b)
@test hash(r1.clause) == hash(r1b.clause)

algo = SIRUS.Classification()
empty_model = S.StableRules(S.Rule[], algo, [1], Float16[0.1])
@test_throws AssertionError S._predict(empty_model, [31000])

@test S._predict(S.StableRules([r1], algo, [1], Float16[1.0]), [31000]) == [0.61]
@test S._predict(S.StableRules([r1], algo, [1], Float16[1.0]), [33000]) == [0.408]
let
    model = S.StableRules([r1, r5], algo, [1], Float16[0.5, 0.5])
    @test S._predict(model, [33000, 0, 61]) == [mean([0.408, 0.56])]
end

function generate_rules()
    algo = S.Classification()
    forest = S._forest(_rng(), algo, X, y)
    rulesmodel = let
        rules = S._rules(forest)
        weights = repeat(Float16[1.0], length(rules))
        S.StableRules(rules, forest.algo, forest.classes, weights)
    end
    model = StableRulesClassifier(; max_rules=10)
    processed = S.StableRules(forest, X, y, model)
    (; forest, rulesmodel, processed)
end

generated = map(i -> generate_rules(), 1:10)

"""
Return whether the score for the model is roughly equal to check whether RNG is used correctly.
Checking the scores is easier than the raw models since those seem to differ slightly (probably
due to multi-threading, which can change the order).
"""
function equal_output(stage::Symbol)
    V = getproperty.(generated, stage)
    P = S._predict.(V, Ref(X))
    A = auc.(P, Ref(y))
    for i in eachindex(A)
        if !(A[i] ≈ A[1])
            return false
        end
    end
    return true
end
@test equal_output(:forest)
@test equal_output(:rulesmodel)
@test equal_output(:processed)

nothing
