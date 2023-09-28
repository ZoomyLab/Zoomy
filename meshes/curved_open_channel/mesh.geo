Point(1) = {0, 0, 0, 1.0};

/* inflow */
Point(2) = {6.04, -4.195, 0, 1.0};
Point(3) = {6.04, -3.125, 0, 1.0};

/* sec 1 */
Point(4) = {2.03, -4.195,-0.003403, 1.0};
Point(5) = {2.03, -3.125,-0.003403, 1.0};

/* sec 2 */
Point(6) = {0.82, -4.195, -0.0044073, 1.0};
Point(7) = {0.82, -3.125, -0.0044073, 1.0};

/* sec 3 */
Point(8) = {0., -4.195, -0.0050879, 1.0};
Point(9) = {0., -3.125, -0.0050879, 1.0};

/* sec 4 */
Point(10) = {-2.0975, -3.63297657, -0.00667849, 1.0};
Point(11) = {-1.5625, -2.70632939, -0.00667849, 1.0};

/* sec 5 */
Point(12) = {-3.63297657, -2.0975, -0.00826908, 1.0};
Point(13) = {-2.70632939, -1.5625, -0.00826908, 1.0};

/* sec 6 */
Point(14) = {-4.195, 0, -0.00985966508, 1.0};
Point(15) = {-3.125, 0, -0.00985966508, 1.0};

/* sec 7 */
Point(16) = {-3.63297657, 2.0975, -0.01145025, 1.0};
Point(17) = {-2.70632939, 1.5625, -0.01145025, 1.0};

/* sec 8 */
Point(18) = {-2.0975, 3.63297657, -0.01304084, 1.0};
Point(19) = {-1.5625, 2.70632939, -0.01304084, 1.0};

/* sec 9 */
Point(20) = {0, 4.195, -0.0146314302, 1.0};
Point(21) = {0, 3.125, -0.0146314302, 1.0};

/* sec 10 */
Point(22) = {2.0975, 3.63297657,  -0.01622202, 1.0};
Point(23) = {1.5625, 2.70632939,  -0.01622202, 1.0};

/* sec 11 */
Point(24) = {3.63297657, 2.0975, -0.01781261, 1.0};
Point(25) = {2.70632939, 1.5625, -0.01781261, 1.0};

/* sec 12 */
Point(26) = {4.195, 0, -0.0194031952, 1.0};
Point(27) = {3.125, 0, -0.0194031952, 1.0};

/* sec 13 */
Point(28) = {4.195, -1.66,  -0.020781, 1.0};
Point(29) = {3.125, -1.66,  -0.020781, 1.0};

/* sec outflow */
Point(30) = {4.195, -2.53, -0.0215031, 1.0};
Point(31) = {3.125, -2.53, -0.0215031, 1.0};

Line(2) = {2,4};
Line(3) = {4,6};
Line(4) = {6,8};
Circle(5) = {8, 1, 10};
Circle(6) = {10, 1, 12};
Circle(7) = {12, 1, 14};
Circle(8) = {14, 1, 16};
Circle(9) = {16, 1, 18};
Circle(10) = {18, 1, 20};
Circle(11) = {20, 1, 22};
Circle(12) = {22, 1, 24};
Circle(13) = {24, 1, 26};
Line(14) = {26, 28};
Line(15) = {28, 30};
Line(17) = {31, 29};
Line(18) = {29, 27};
Circle(19) = {27,1,25};
Circle(20) = {25,1,23};
Circle(21) = {23,1,21};
Circle(22) = {21,1,19};
Circle(23) = {19,1,17};
Circle(24) = {17,1,15};
Circle(25) = {15,1,13};
Circle(26) = {13,1,11};
Circle(27) = {11,1,9};
Line(28) = {9,7};
Line(29) = {7,5};
Line(30) = {5,3};

/* crossings */
Line(1) = {2,3};
Line(31) = {5,4};
Line(32) = {7,6};
Line(33) = {9,8};
Line(34) = {11,10};
Line(35) = {13,12};
Line(36) = {15,14};
Line(37) = {17,16};
Line(38) = {19,18};
Line(39) = {21,20};
Line(40) = {23,22};
Line(41) = {25,24};
Line(42) = {27,26};
Line(43) = {29,28};
Line(16) = {31, 30};

b = 1.07/2;
l01 = 4.1;
l12 = 1.21;
l23 = 0.82;
l1213 = 1.66;
l1314 = 0.87;
 // arclength circa 3
arclength = 30*3.14*(3.66-b)/180;

res = 5;
res_rad = res*2*b+1;
res_arc = res*arclength;
res_l01 = res*4.1;
res_l12 = res*1.21;
res_l23 = res*0.82;
res_l1213 = res*1.66;
res_l1314 = res*0.87;


Curve Loop(1) = {1, -2, 31 ,-30 };
Surface(1) = {-1};
Transfinite Line {1, 31} = res_rad ;
Transfinite Line {-2, 30} = res_l01;
Transfinite Surface {1};

Curve Loop(2) = {3, -32 , 29 , 31 };
Surface(2) = {2};
Transfinite Line {3, 29} = res_l12;
Transfinite Line {32, 31} = res_rad;
Transfinite Surface {2};

Curve Loop(3) = {4, -33 , 28 , 32 };
Surface(3) = {3};
Transfinite Line {4, 28} = res_l23;
Transfinite Line {33, 32} = res_rad;
Transfinite Surface {3};

Curve Loop(4) = {5, -34 , 27 , 33 };
Surface(4) = {4};
Transfinite Line {5,27} = res_arc;
Transfinite Line { -34, 33} = res_rad;
Transfinite Surface {4};

Curve Loop(5) = {6, -35 , 26 , 34 };
Surface(5) = {5};
Transfinite Line { 34, 35} = res_rad;
Transfinite Line {26,6} = res_arc;
Transfinite Surface {5};

Curve Loop(6) = {7, -36 , 25 , 35 };
Surface(6) = {6};
Transfinite Line { 35, -36} = res_rad;
Transfinite Line {25,7} = res_arc;
Transfinite Surface {6};

Curve Loop(7) = {8, -37 , 24 , 36 };
Surface(7) = {7};
Transfinite Line { 36, -37} = res_rad;
Transfinite Line {24,8} = res_arc;
Transfinite Surface {7};

Curve Loop(8) = {9, -38 , 23 , 37 };
Surface(8) = {8};
Transfinite Line { 37, -38} = res_rad;
Transfinite Line {23,9} = res_arc;
Transfinite Surface {8};

Curve Loop(9) = {10, -39 , 22 , 38 };
Surface(9) = {9};
Transfinite Line { 38, -39} = res_rad;
Transfinite Line {22,10} = res_arc;
Transfinite Surface {9};

Curve Loop(10) = {11, -40 , 21 , 39 };
Surface(10) = {10};
Transfinite Line { 39, -40} = res_rad;
Transfinite Line {21,11} = res_arc;
Transfinite Surface {10};

Curve Loop(11) = {12, -41 , 20 , 40 };
Surface(11) = {11};
Transfinite Line { 40, -41} = res_rad;
Transfinite Line {20,12} = res_arc;
Transfinite Surface {11};

Curve Loop(12) = {13, -42 , 19 , 41 };
Surface(12) = {12};
Transfinite Line { 41, -42} = res_rad;
Transfinite Line {19,13} = res_arc;
Transfinite Surface {12};

Curve Loop(13) = {14, -43 , 18 , 42 };
Surface(13) = {13};
Transfinite Line {14, 18} = res_l1213;
Transfinite Line {43, 42} = res_rad;
Transfinite Surface {13};

Curve Loop(14) = {-16, 17 , 43 , 15 };
Surface(14) = {14};
Transfinite Line {15, 17} = res_l1314;
Transfinite Line {43, 16} = res_rad;
Transfinite Surface {14};

Recombine Surface "*";

// Physical Curve("wall") = {14, 18};
Physical Curve("inflow", 1000) = {1};
Physical Curve("outflow", 1001) = {16};
Physical Curve("wall", 1002) = {2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21,22,23,24,25,26,27,28,29,30};
Physical Surface("surface", 2000) = {1,2,3,4,5,6,7,8,9,10,11,12,13,14};

// Physical Curve("innerwall") = {29, -28};
// Physical Curve("outerwall") = {3, 4};
// Physical Curve("inflow") = {2};
// Physical Curve("outflow") = {16};
// Physical Surface("surface") = {};
