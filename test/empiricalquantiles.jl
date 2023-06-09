X = [1 2;
     3 4]
y = [1, 2]

@test cutpoints([3, 1, 2], 2) == Float32[1, 2]
@test cutpoints(1:9, 3) == Float32[3, 5, 7]
@test cutpoints(1:4, 3) == Float32[1, 2, 3]
@test cutpoints([1, 3, 5, 7], 2) == Float32[3, 5]

@test cutpoints(X, 2) == [Float32[1, 3], Float32[2, 4]]
@test cutpoints([3 4; 1 5; 2 6], 2) == [Float32[1, 2], Float32[4, 5]]
