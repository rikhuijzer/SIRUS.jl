text = " X[i, 1] < 1.0 & X[i, 1] ≥ 4.0 "
@test repr(TreePath(text)) == "TreePath(\"$text\")"

Float = ST.Float
classes = [:a, :b, :c]
left = ST.Leaf([1.0, 0.0, 0.0])
splitpoint = ST.SplitPoint(1, Float(1))
right = ST.Node(splitpoint, ST.Leaf([0.0, 1.0, 0.0]), ST.Leaf([0.0, 0.0, 1.0]))

left_rule = ST.Rule(ST.TreePath(" X[i, 1] < 32000 "), [0.61], [0.408])
right_rule = ST.Rule(ST.TreePath(" X[i, 1] ≥ 32000 "), [0.408], [0.61])
@test ST._flip_left([left_rule, right_rule]) == [left_rule, left_rule]

r1 = ST.Rule(ST.TreePath(" X[i, 1] < 32000 "), [0.61], [0.408])
r1b = ST.Rule(ST.TreePath(" X[i, 1] < 32000 "), [0.61], [0.408])
r1c = ST.Rule(ST.TreePath(" X[i, 1] < 32000 "), [0.0], [0.408])
r5 = ST.Rule(ST.TreePath(" X[i, 3] < 64 "), [0.56], [0.334])

let
    expected = [r1 => 2, r5 => 1]
    @test ST._combine_paths(ST._flip_left([r5, r1, r1])) == expected
end
@test ST._mean([[1, 4], [2, 4]]) == [1.5, 4.0]

# @test ST._mode([[1, 2], [1, 6], [4, 6]]) == [1, 6]

splitpoint = ST.SplitPoint(1, ST.Float(4))
node = ST.Node(splitpoint, left, right)

rules = ST._rules!(node)

n = 200
X, y = make_moons(n; rng=_rng(), shuffle=true)
model = StableForestClassifier(; rng=_rng())
mach = machine(model, X, y)
fit!(mach; verbosity=0)
forest = mach.fitresult

rules = ST._rules(forest)

@test hash(r1) == hash(r1b)
@test hash(r1.path) == hash(r1b.path)

@test ST._combine_paths([r1, r1b]) == [r1 => 2]
@test first(only(ST._combine_paths([r1, r1c]))).then_probs == [mean([0.61, 0])]
@test ST._count_unique([1, 1, 1, 2]) == Dict(1 => 3, 2 => 1)

weights = [0.395, 0.197, 0.187, 0.057, 0.054, 0.043, 0.027, 0.02, 0.01, 0.01]
regularized = ST._regularize_weights(weights)
@test regularized[1] < weights[1]
@test regularized[end] > weights[end]

empty_model = ST.StableRules(ST.Rule[], [1], [0.1])
@test_throws AssertionError ST._predict(empty_model, [31000])

@test ST._predict(ST.StableRules([r1], [1], [1.0]), [31000]) == [0.61]
@test ST._predict(ST.StableRules([r1], [1], [1.0]), [33000]) == [0.408]
let
    model = ST.StableRules([r1, r5], [1], [0.5, 0.5])
    @test ST._predict(model, [33000, 0, 61]) == [mean([0.408, 0.56])]
end

@test first(ST._process_rules([r5, r1, r1], 10)) == Pair(r1, 2)

let
    model = ST.StableRules([r5, r1, r1], [1], 10)
    @test model.rules == [r1, r5]
    @test model.classes == [1]
    @test model.weights == ST._regularize_weights([2/3, 1/3])
end

function generate_rules()
    forest = ST._forest(_rng(), X, y)
    max_classes = 10
    rulesmodel = let
        rules = ST._rules(forest)
        weights = repeat([1.0], length(rules))
        ST.StableRules(rules, forest.classes, weights)
    end
    processed = ST.StableRules(forest, max_classes)
    (; forest, rulesmodel, processed)
end

generated = map(i -> generate_rules(), 1:10)

"""
Return whether the score for the model is roughly equal to check whether RNG is used correctly.
Checking the scores is easier than the raw models since those seem to differ slightly.
"""
function equal_output(stage::Symbol)
    V = getproperty.(generated, stage)
    P = ST._predict.(V, Ref(X))
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
