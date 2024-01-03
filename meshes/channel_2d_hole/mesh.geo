length = 8.;
width =  2.;
x_hole = length/4;
y_hole = width/2;
r_hole = 0.2;
d_hole = r_hole / Sqrt(2);



Point(1) = {0, 0, 0, 1.0};
Point(2) = {0, width, 0, 1.0};
Point(3) = {length, width, 0, 1.0};
Point(4) = {length, 0, 0, 1.0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};


Point(5) = {x_hole-d_hole  , y_hole-d_hole, 0, 1.0};
Point(6) = {x_hole-d_hole  , y_hole+d_hole, 0, 1.0};
Point(7) = {x_hole+d_hole  , y_hole+d_hole, 0, 1.0};
Point(8) = {x_hole+d_hole  , y_hole-d_hole, 0, 1.0};
Point(9) = {x_hole         , y_hole, 0, 1.0};

Circle(5) = {5, 9, 6};
Circle(6) = {6, 9, 7};
Circle(7) = {7, 9, 8};
Circle(8) = {8, 9, 5};

Line(10) = {1, 5};
Line(11) = {2, 6};
Line(12) = {3, 7};
Line(13) = {4, 8};

Curve Loop(5) = {1,2,3,4};
Curve Loop(1) = {1};
Curve Loop(2) = {2};
Curve Loop(3) = {3};
Curve Loop(4) = {4};
/* Curve Loop(1) = {1,2,3,4}; */
/* Curve Loop(1) = {1,2,3,4}; */
Curve Loop(6) = {8,7,6,5};
Surface(1) = {5,6};

/* Transfinite Surface {1,2,3,4}; */


Physical Curve("bottom", 1000) = {2};
Physical Curve("top", 1001) = {4};
Physical Curve("right", 1002) = {3};
Physical Curve("left", 1003) = {1};
Physical Curve("hole", 1004) = {5,6,7,8};
Physical Surface("surface", 2000) = {1, 2, 3, 4};

