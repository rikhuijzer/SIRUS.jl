using StableTrees
using Test

const ST = StableTrees
const Float = ST.Float

@test ST.gini([1, 1, 1], [1]) == Float(0.0)

feature = 1
@test collect(ST._view_y([1 2; 3 4], [1 2], feature, <, 2)) == [1]
@test collect(ST._view_y([1 2; 3 4], [1 2], feature, >, 2)) == [2]

@test ST._cutpoints([3, 1, 2], 2) == Float[1, 3]
@test ST._cutpoints(1:10, 3) == Float[1, 5, 10]

@test ST._cutpoints([1 2; 3 4], 2) == Float[1 2; 3 4]
@test ST._cutpoints([3 4; 1 5; 2 6], 2) == Float[1 4; 3 6]

X = [1 2; 3 4]
y = [1, 2]
classes = [1, 2]
cutpoints = Float[1 3; 4 6]
# @test ST._find_split([1 2; 3 4], vec([1 2]), vec([1 2])) == (1, 3)
