text = " X[i, 1] < 1.0 & X[i, 1] ≥ 4.0 "
@test repr(TreePath(text)) == "TreePath(\"$text\")"

Float = ST.Float
classes = [:a, :b, :c]
left = ST.Leaf([1.0, 0.0, 0.0])
splitpoint = ST.SplitPoint(1, Float(1))
right = ST.Node(splitpoint, ST.Leaf([0.0, 1.0, 0.0]), ST.Leaf([0.0, 0.0, 1.0]))

@test ST._frequency_sort([1, 2, 2, 2, 3, 3]) == [2, 3, 1]
@test ST._mean_probabilities([[1, 4], [2, 4]]) == [1.5, 4.0]

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

r1 = ST.Rule(ST.TreePath(" X[i, 1] < 32000 "), [0.61], [0.408])
r1b = ST.Rule(ST.TreePath(" X[i, 1] < 32000 "), [0.61], [0.408])

@test hash(r1) == hash(r1b)
@test ST._count_unique([1, 1, 1, 2]) == Dict(1 => 3, 2 => 1)

r5 = ST.Rule(ST.TreePath(" X[i, 3] < 64 "), [0.56], [0.334])

@test ST._predict(ST.StableRules([r1], [1]), [31000]) == [0.61]
@test ST._predict(ST.StableRules([r1], [1]), [33000]) == [0.408]
let
    model = ST.StableRules([r1, r5], [1])
    @test ST._predict(model, [33000, 0, 61]) == [mean([0.408, 0.56])]
end

function generate_rules()
    forest = ST._forest(_rng(), X, y)
    _model(rules::Vector{ST.Rule}) = ST.StableRules(rules, forest.classes)
    rules = _model(ST._rules(forest))
    processed = _model(ST._process_rules(ST._elements(rules), 10))
    (; forest, rules, processed)
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
@test equal_output(:rules)
@test equal_output(:processed)
