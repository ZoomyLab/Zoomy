length = 4.;
width =  2.;
x_hole = length/8;
y_hole = width/2;
r_hole = 0.2;
d_hole = r_hole / Sqrt(2);

/* scale = 0.5; */
/* scale = 0.25; */
/* scale = 0.125; */
/* scale = 0.0625; */
scale = 0.03125;

Point(1) = {0, 0, 0, scale};
Point(2) = {0, y_hole, 0, scale};
Point(3) = {x_hole-r_hole, y_hole, 0, scale};
Point(4) = {x_hole-d_hole, y_hole-d_hole, 0, scale};
Point(5) = {x_hole+d_hole, y_hole-d_hole, 0, scale};
Point(6) = {x_hole+r_hole, y_hole, 0, scale};
Point(7) = {length, y_hole, 0, scale};
Point(8) = {length, 0, 0, scale};
Point(9) = {x_hole, y_hole, 0, scale};
Point(51) = {0, width, 0, scale};
/* Point(52) = {0, y_hole, 0, scale}; */
/* Point(53) = {x_hole-r_hole, y_hole, 0, scale}; */
Point(54) = {x_hole-d_hole, y_hole+d_hole, 0, scale};
Point(55) = {x_hole+d_hole, y_hole+d_hole, 0, scale};
/* Point(56) = {x_hole+r_hole, y_hole, 0, scale}; */
/* Point(57) = {length, y_hole, 0, scale}; */
Point(58) = {length, width, 0, scale};
/* Point(59) = {x_hole, y_hole, 0, scale}; */

Line(1) = {1, 2};
Line(2) = {2, 3};
Circle(3) = {3, 9, 4};
Circle(4) = {4, 9, 5};
Circle(5) = {5, 9, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 1};
Line(9) = {4,1};
Line(10) = {8,5};

Line(51) = {51, 2};
/* Line(52) = {2, 3}; */
Circle(53) = {3, 9, 54};
Circle(54) = {54, 9, 55};
Circle(55) = {55, 9, 6};
/* Line(56) = {6, 7}; */
Line(57) = {7, 58};
Line(58) = {58, 51};
Line(59) = {54,51};
Line(60) = {58,55};

Curve Loop(100) = {1,2,3,9};
Curve Loop(101) = {-9, 4, -10, 8};
Curve Loop(102) = {5,6,7,10};
Surface(1001) = {101};
Surface(1000) = {100};
Surface(1002) = {102};

Curve Loop(150) = {-51,-2,-53,-59};
Curve Loop(151) = {59, -54, 60, -58};
Curve Loop(152) = {-55,-6,-57,-60};
Surface(1051) = {151};
Surface(1050) = {150};
Surface(1052) = {152};
/* Surface(1000) = {100, 101, 102 ,150, 151, 152}; */

/* Transfinite Surface {1000}; */


Physical Curve("bottom", 10000) = {8};
Physical Curve("top", 10001) = {58};
Physical Curve("right", 10002) = {7, 57};
Physical Curve("left", 10003) = {1, 51};
Physical Curve("hole", 10004) = {3,4,5,53,54,55};
Physical Surface("surface", 20000) = {1000, 1001, 1002, 1050, 1051, 1052};

