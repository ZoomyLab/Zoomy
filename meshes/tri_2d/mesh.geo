//+
Point(1) = {-1, -1, 0, 1.0};
//+
Point(2) = {1, -1, 0, 1.0};
//+
Point(3) = {-1, 1, 0, 1.0};
//+
Point(4) = {1, 1, 0, 1.0};

//+
Line(1) = {3, 4};
//+
Line(2) = {4, 2};
//+
Line(3) = {2, 1};
//+
Line(4) = {1, 3};
//+
Curve Loop(5) = {4, 1, 2, 3};
Plane Surface(6) = {5};

Transfinite Surface {6};

Physical Curve("right", 1000) = {2};
Physical Curve("left", 1001) = {4};
Physical Curve("bottom", 1002) = {3};
Physical Curve("top", 1003) = {1};
Physical Surface("surface", 2000) = {6};
