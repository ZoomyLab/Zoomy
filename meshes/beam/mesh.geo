scale = 0.1;

Point(10) = {0, 0,  0, scale};
Point(11) = {2, 0,  0, scale};
Point(12) = {4, 0,  0, scale};
Point(13) = {6, 0,  0, scale};
Point(14) = {8, 0,  0, scale};
Point(15) = {10, 0,  0, scale};

Point(20) = {0, 1, 0, scale};
Point(21) = {2, 1, 0, scale};
Point(22) = {4, 1, 0, scale};
Point(23) = {6, 1, 0, scale};
Point(24) = {8, 1, 0, scale};
Point(25) = {10, 1, 0, scale};


Line(111) = {10, 11};
Line(112) = {11, 12};
Line(113) = {12, 13};
Line(114) = {13, 14};
Line(115) = {14, 15};

Line(121) = {20, 21};
Line(122) = {21, 22};
Line(123) = {22, 23};
Line(124) = {23, 24};
Line(125) = {24, 25};

Line(130) = {10, 20};
Line(131) = {11, 21};
Line(132) = {12, 22};
Line(133) = {13, 23};
Line(134) = {14, 24};
Line(135) = {15, 25};

Curve Loop(1000) = {130, 121, -131, -111};
Curve Loop(1001) = {131, 122, -132, -112};
Curve Loop(1002) = {132, 123, -133, -113};
Curve Loop(1003) = {133, 124, -134, -114};
Curve Loop(1004) = {134, 125, -135, -115};
Plane Surface(2000) = {1000};
Plane Surface(2001) = {1001};
Plane Surface(2002) = {1002};
Plane Surface(2003) = {1003};
Plane Surface(2004) = {1004};

Physical Curve("left", 3000) = {130};
Physical Curve("right", 3001) = {135};
Physical Curve("bottom_0", 3002) = {111};
Physical Curve("bottom_1", 3003) = {112};
Physical Curve("bottom_2", 3004) = {113};
Physical Curve("bottom_3", 3005) = {114};
Physical Curve("bottom_4", 3006) = {115};

Physical Curve("top_0", 3007) = {121};
Physical Curve("top_1", 3008) = {122};
Physical Curve("top_2", 3009) = {123};
Physical Curve("top_3", 3010) = {124};
Physical Curve("top_4", 3011) = {125};
Physical Surface("surface", 4000) = {2000, 2001, 2002, 2003, 2004};
