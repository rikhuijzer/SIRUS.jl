using StableTrees
using Test

const ST = StableTrees
const Float = ST.Float

@test ST.gini([1, 1, 1], [1]) == 0.0f0

feature = 1
@test collect(ST._view_y([1 2; 3 4], [1 2], feature, <, 2)) == [1]
@test collect(ST._view_y([1 2; 3 4], [1 2], feature, >, 2)) == [2]

@test ST._find_split([1 2; 3 4], vec([1 2]), vec([1 2])) == (1, 3)
