l_1 = 0.3;
l_2 = 0.1;
l_3 = 0.6;
l_junc = 0.6;
angle = 45 * Pi / 180;
width =  0.14;

/* scale = 0.02; */
scale = 0.005;
/*scale = 0.0025; */
/* scale = 0.00125; */

// section 1
Point(1) = {0, 0, 0, scale};
Point(2) = {0, width, 0, scale};
Point(3) = {l_1, width, 0, scale};
Point(4) = {l_1, 0, 0, scale};

Line(11) = {1, 2};
Line(12) = {2, 3};
Line(13) = {3, 4};
Line(14) = {4, 1};
Curve Loop(101) = {11, 12, 13, 14};

// section 2
Point(5) = {l_1 + l_2, 0, 0, scale};
Point(6) = {l_1 + l_2, width, 0, scale};

Line(21) = {4, 5};
Line(22) = {5, 6};
Line(23) = {6, 3};
Curve Loop(102) = {-13, -21, -22, -23};

// section 3
Point(7) = {l_1 + l_2 + l_3, 0, 0, scale};
Point(8) = {l_1 + l_2 + l_3, width, 0, scale};

Line(31) = {5, 7};
Line(32) = {7, 8};
Line(33) = {8, 6};
Curve Loop(103) = {22, -33, -32, -31};

// section junction
Point(9) = {l_1 + Cos(angle)*l_junc, - Sin(angle)*l_junc, 0, scale};
Point(10) = {l_1 + l_2 + Cos(angle)*l_junc, -Sin(angle)*l_junc, 0, scale};

Line(41) = {4, 9};
Line(42) = {9, 10};
Line(43) = {10, 5};
Curve Loop(104) = {21, -41, -42, -43};
/**/
Transfinite Line {1, 4} = scale ;
Transfinite Line {4, 5} = scale ;
Transfinite Line {5, 7} = scale ;

Surface(1001) = {101};
Transfinite Surface {1001};

Surface(1002) = {102};
Transfinite Surface {1002};

Surface(1003) = {103};
Transfinite Surface {1003};

Surface(1004) = {104};
Transfinite Surface {1004};


Physical Curve("inflow") = {11};
Physical Curve("wall") = {12, 23, 33, 32, 31, 43, 42, 41, 14};
Physical Surface("volume") = {1001, 1002, 1003, 1004};
