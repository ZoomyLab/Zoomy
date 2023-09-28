Point(1) = {0, 0, 0, 1.0};
Point(2) = {1, 0, 0, 1.0};
Point(3) = {1, 1, 0, 1.0};
Point(4) = {0, 1, 0, 1.0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};


Point(5) = {0.4, 0.4, 0, 1.0};
Point(6) = {0.6, 0.4, 0, 1.0};
Point(7) = {0.6, 0.6, 0, 1.0};
Point(8) = {0.4, 0.6, 0, 1.0};
Point(9) = {0.5, 0.5, 0, 1.0};

Circle(5) = {5, 9, 6};
Circle(6) = {6, 9, 7};
Circle(7) = {7, 9, 8};
Circle(8) = {8, 9, 5};

Line(10) = {1, 5};
Line(11) = {2, 6};
Line(12) = {3, 7};
Line(13) = {4, 8};

Curve Loop(1) = {10, 5, -11, -1};
Surface(1) = {1};
Curve Loop(2) = {11, 6, -12, -2};
Surface(2) = {2};
Curve Loop(3) = {12, 7, -13, -3};
Surface(3) = {3};
Curve Loop(4) = {13, 8, -10, -4};
Surface(4) = {4};
Curve Loop(5) = {5,8,7,6};
/* Surface(5) = {5}; */

Transfinite Surface {1,2,3,4};
Recombine Surface {1,2,3,4};

Physical Curve("right", 1000) = {2};
Physical Curve("left", 1001) = {4};
Physical Curve("top", 1002) = {3};
Physical Curve("bottom", 1003) = {1};
Physical Curve("hole", 1004) = {5,6,7,8};
Physical Surface("surface", 2000) = {1, 2, 3, 4};

