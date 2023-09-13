rng = StableRNG(1)

@testset "_copy_rng makes independent copies" begin
    expected = rand(rng)
    expected2 = rand(rng)
    for i in 1:50
        _rng = SIRUS._copy_rng(rng)
        seed!(_rng, 1)
        @test rand(_rng) == expected
        @test rand(_rng) == expected2
    end
end

X = [1 2;
     3 4]
y = [1, 2]

y_view = Vector{eltype(y)}(undef, length(y))
feature = 1
@test collect(S._view_y!(y_view, X[:, feature], [1 2], <, 2)) == [1]
@test collect(S._view_y!(y_view, X[:, feature], [1 2], >, 2)) == [2]

for algo in [SIRUS.Classification(), SIRUS.Regression()]
    is_classification = algo isa SIRUS.Classification
    xs = [1 1;
         1 3]
    ys = [1.0, 2.0]
    cs = is_classification ?  unique(ys) : []
    colnms = ["A", "B"]
    cp = cutpoints(xs, 2)
    max_split_candidates::Int = SIRUS.nfeatures(xs)
    sp = S._split(StableRNG(1), algo, xs, ys, cs, colnms, cp; max_split_candidates)
    @test !isnothing(sp)
    # Obviously, feature (column) 2 is more informative to split on than feature 1.
    @test sp.feature == 2
    @test sp.feature_name == "B"
    # Given that the split does < and ≥, then 3 is the best place since it separates 1 (left) and 3 (right).
    @test sp.value == Float32(3)
end

nothing
