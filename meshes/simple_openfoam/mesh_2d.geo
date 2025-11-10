length = 4.;
width =  2.;
x_hole = length/8;
y_hole = width/2;
r_hole = 0.2;
d_hole = r_hole / Sqrt(2);
height_water = 0.8;
height_air = 0.2;
N_layers = 30;
N_layers_water = N_layers * height_water/(height_air + height_water);
N_layers_air = N_layers * height_air/(height_air + height_water);

/* scale = 0.5; */
/* scale = 0.25; */
scale = 0.125;
/* scale = 0.0625; */
/* scale = 0.03125; */

Point(1) = {0, 0, 0, scale};
Point(2) = {0, width, 0, scale};
Point(3) = {length, width, 0, scale};
Point(4) = {length, 0, 0, scale};
/* Point(5) = {x_hole-r_hole, y_hole, 0, scale}; */
/* Point(6) = {x_hole, y_hole+r_hole, 0, scale}; */
/* Point(7) = {x_hole+r_hole, y_hole, 0, scale}; */
/* Point(8) = {x_hole, y_hole-r_hole, 0, scale}; */
/* Point(9) = {x_hole, y_hole, 0, scale}; */

Line(11) = {1, 2};
Line(12) = {2, 3};
Line(13) = {3, 4};
Line(14) = {4, 1};
/* Circle(15) = {5, 9, 6}; */
/* Circle(16) = {6, 9, 7}; */
/* Circle(17) = {7, 9, 8}; */
/* Circle(18) = {8, 9, 5}; */

Curve Loop(101) = {11,12,13,14};
/* Curve Loop(102) = {15, 16, 17, 18}; */
Surface(1001) = {101};

Physical Curve("inflow") = {11};
Physical Curve("top") = {12};
Physical Curve("outflow") = {13};
Physical Curve("bottom") = {14};
/* Physical Curve("pillar") = {16, 17, 18, 15}; */
Physical Surface("volume") = {1001};
//+
