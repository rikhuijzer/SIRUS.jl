using StableTrees
using Test

const ST = StableTrees
const Float = ST.Float

@test ST.gini([1, 1, 1], [1]) == 0.0f0

feature = 1
@test collect(ST._view_y([1 2; 3 4], [1 2], feature, <, 2)) == [1]
@test collect(ST._view_y([1 2; 3 4], [1 2], feature, >, 2)) == [2]

@test ST._cutpoints([3, 1, 2], 2) == [1, 3]
@test ST._cutpoints(1:10, 3) == [1, 5, 10]

@test ST._find_split([1 2; 3 4], vec([1 2]), vec([1 2])) == (1, 3)
