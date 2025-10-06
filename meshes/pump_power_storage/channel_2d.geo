alpha = 45;
s_a = Sin(alpha * Pi / 180);
c_a = Cos(alpha * Pi / 180);

//l_side = 1.45;
dx_width_channel = 0.1; 
dx_width_wall = 0.025;
dx_width_gap = 0.3;
dx_offset_channel = 2 * dx_width_wall + dx_width_gap;
l_inflow = 0.15;
dx_width_front = 0.1;
dx_width_back = 0.1;
l_channel = 2.09 - dx_width_front - dx_width_back;

L = 2.;
z_A = 0.0007 * L;
z_B = 0.0008 * L;
z_C = 0.0009 * L;
z_D = 0.0010 * L;

z_A_end = 0.0018 * L;
z_B_end = 0.0018 * L;
z_C_end = 0.0017 * L;
z_D_end = 0.0017 * L;

bump_base = 1/4;
bump_inlet_x = bump_base/1;
bump_inlet_y = bump_base/1;
bump_square_x = bump_base/1;
bump_square_y = bump_base/1;
bump_gap_x = bump_base/1;
bump_gap_y = bump_base/1;
bump_channel_x = bump_base/8;
bump_channel_y = bump_base/1;


/* scale = 0.5; */
/* scale = 0.25; */
/* scale = 0.125; */
/* scale = 0.0625; */
scale = 0.03125;

transfinite_scale = 10;
res_coarse = transfinite_scale * 3.;
res = res_coarse * 10.;
res_fine = res * 1.0;

// offset to merge the inflow to the outflow of the inlet
d_off = 0.1 * Sqrt(2)/2;

// First 
P1[] = {c_a * l_inflow, s_a * l_inflow};
P2[] = {P1[0] + s_a*dx_width_channel, P1[1]+c_a*dx_width_channel};
P3[] = {P2[0] + s_a*dx_offset_channel, P2[1]+c_a*dx_offset_channel};
P4[] = {P3[0] + s_a*dx_width_channel, P3[1]+c_a*dx_width_channel};
P5[] = {P4[0] + s_a*dx_offset_channel, P4[1]+c_a*dx_offset_channel};
P6[] = {P5[0] + s_a*dx_width_channel, P5[1]+c_a*dx_width_channel};
P7[] = {P6[0] + s_a*dx_offset_channel, P6[1]+c_a*dx_offset_channel};
P8[] = {P7[0] + s_a*dx_width_channel, P7[1]+c_a*dx_width_channel};

Point(1) = {P1[0], P1[1], 0, scale};
Point(2) = {P2[0], P2[1], 0, scale};
Point(3) = {P3[0], P3[1], 0, scale};
Point(4) = {P4[0], P4[1], 0, scale};
Point(5) = {P5[0], P5[1], 0, scale};
Point(6) = {P6[0], P6[1], 0, scale};
Point(7) = {P7[0], P7[1], 0, scale};
Point(8) = {P8[0], P8[1], 0, scale};

Line(12) = {1, 2};
Line(23) = {2, 3};
Line(34) = {3, 4};
Line(45) = {4, 5};
Line(56) = {5, 6};
Line(67) = {6, 7};
Line(78) = {7, 8};

// Second
P11[] = {P1[0] + dx_width_front, P1[1]};
P12[] = {P2[0] + dx_width_front, P2[1]};
P13[] = {P3[0] + dx_width_front, P3[1]};
P14[] = {P4[0] + dx_width_front, P4[1]};
P15[] = {P5[0] + dx_width_front, P5[1]};
P16[] = {P6[0] + dx_width_front, P6[1]};
P17[] = {P7[0] + dx_width_front, P7[1]};
P18[] = {P8[0] + dx_width_front, P8[1]};

Point(11) = {P11[0], P11[1], 0, scale};
Point(12) = {P12[0], P12[1], 0, scale};
Point(13) = {P13[0], P13[1], 0, scale}; 
Point(14) = {P14[0], P14[1], 0, scale};
Point(15) = {P15[0], P15[1], 0, scale};
Point(16) = {P16[0], P16[1], 0, scale};
Point(17) = {P17[0], P17[1], 0, scale};
Point(18) = {P18[0], P18[1], 0, scale};

Line(112) = {11, 12};
Line(123) = {12, 13};
Line(134) = {13, 14};
Line(145) = {14, 15};
Line(156) = {15, 16};
Line(167) = {16, 17};
Line(178) = {17, 18};

// Third
P21[] = {P11[0] + l_channel, P11[1]};
P22[] = {P12[0] + l_channel, P12[1]};
P23[] = {P13[0] + l_channel, P13[1]};
P24[] = {P14[0] + l_channel, P14[1]};
P25[] = {P15[0] + l_channel, P15[1]};
P26[] = {P16[0] + l_channel, P16[1]};
P27[] = {P17[0] + l_channel, P17[1]};
P28[] = {P18[0] + l_channel, P18[1]};

Point(21) = {P21[0], P21[1], 0, scale};
Point(22) = {P22[0], P22[1], 0, scale};
Point(23) = {P23[0], P23[1], 0, scale};
Point(24) = {P24[0], P24[1], 0, scale};
Point(25) = {P25[0], P25[1], 0, scale};
Point(26) = {P26[0], P26[1], 0, scale};
Point(27) = {P27[0], P27[1], 0, scale};
Point(28) = {P28[0], P28[1], 0, scale};

Line(212) = {21, 22};
Line(223) = {22, 23};
Line(234) = {23, 24};
Line(245) = {24, 25};
Line(256) = {25, 26};
Line(267) = {26, 27};
Line(278) = {27, 28};

// Fourth
P31[] = {P21[0] + dx_width_back, P21[1]};
P32[] = {P22[0] + dx_width_back, P22[1]};
P33[] = {P23[0] + dx_width_back, P23[1]};
P34[] = {P24[0] + dx_width_back, P24[1]};
P35[] = {P25[0] + dx_width_back, P25[1]};
P36[] = {P26[0] + dx_width_back, P26[1]};
P37[] = {P27[0] + dx_width_back, P27[1]};
P38[] = {P28[0] + dx_width_back, P28[1]};

Point(31) = {P31[0], P31[1], 0, scale};
Point(32) = {P32[0], P32[1], 0, scale};
Point(33) = {P33[0], P33[1], 0, scale};
Point(34) = {P34[0], P34[1], 0, scale};
Point(35) = {P35[0], P35[1], 0, scale};
Point(36) = {P36[0], P36[1], 0, scale};
Point(37) = {P37[0], P37[1], 0, scale};
Point(38) = {P38[0], P38[1], 0, scale};

Line(312) = {31, 32};
Line(323) = {32, 33};
Line(334) = {33, 34};
Line(345) = {34, 35};
Line(356) = {35, 36};
Line(367) = {36, 37};
Line(378) = {37, 38};

// Horizontals

Line(1001) = { 1, 11};
Line(1011) = {11, 21};
Line(1021) = {21, 31};

Line(1002) = { 2, 12};
Line(1012) = {12, 22};
Line(1022) = {22, 32};

Line(1003) = { 3, 13};
Line(1013) = {13, 23};
Line(1023) = {23, 33};

Line(1004) = { 4, 14};
Line(1014) = {14, 24};
Line(1024) = {24, 34};

Line(1005) = { 5, 15};
Line(1015) = {15, 25};
Line(1025) = {25, 35};

Line(1006) = { 6, 16};
Line(1016) = {16, 26};
Line(1026) = {26, 36};

Line(1007) = { 7, 17};
Line(1017) = {17, 27};
Line(1027) = {27, 37};

Line(1008) = { 8, 18};
Line(1018) = {18, 28};
Line(1028) = {28, 38};

// Curve loops
// Note: Transfinite Line takes the parallel lines as input

// first column
Curve Loop(10001) = {12, 1002, -112 , -1001};
Surface(20001) = {10001};
Transfinite Line {12}   = res_fine * dx_width_channel Using Bump bump_square_y;
Transfinite Line {112}   = res_fine * dx_width_channel Using Bump bump_square_y;
Transfinite Line {1001, 1002}  = res * dx_width_channel Using Bump bump_square_x;
Transfinite Surface {20001};

Curve Loop(10002) = {23, 1003, -123 , -1002};
Surface(20002) = {10002};
Transfinite Line {23}   = res * dx_width_gap Using Bump bump_gap_y;
Transfinite Line {123}   = res * dx_width_gap Using Bump bump_gap_y;
Transfinite Line {1003}  = res * dx_width_channel Using Bump bump_gap_x;
Transfinite Surface {20002};

Curve Loop(10003) = {34, 1004, -134 , -1003};
Surface(20003) = {10003};
Transfinite Line {34}   = res * dx_width_channel Using Bump bump_square_y;
Transfinite Line {134}   = res * dx_width_channel Using Bump bump_square_y;
Transfinite Line {1004}  = res * dx_width_channel Using Bump bump_square_x;
Transfinite Surface {20003};

Curve Loop(10004) = {45, 1005, -145 , -1004};
Surface(20004) = {10004};
Transfinite Line {45}   = res * dx_width_gap Using Bump bump_gap_y;
Transfinite Line {145}   = res * dx_width_gap Using Bump bump_gap_y;
Transfinite Line {1005}  = res * dx_width_channel Using Bump bump_gap_x;
Transfinite Surface {20004};

Curve Loop(10005) = {56, 1006, -156 , -1005};
Surface(20005) = {10005};
Transfinite Line {56}   = res * dx_width_channel Using Bump bump_square_y;
Transfinite Line {156}   = res * dx_width_channel Using Bump bump_square_y;
Transfinite Line {1006}  = res * dx_width_channel Using Bump bump_square_x;
Transfinite Surface {20005};

Curve Loop(10006) = {67, 1007, -167 , -1006};
Surface(20006) = {10006};
Transfinite Line {67}   = res * dx_width_gap Using Bump bump_gap_y;
Transfinite Line {167}   = res * dx_width_gap Using Bump bump_gap_y;
Transfinite Line {1007}  = res * dx_width_channel Using Bump bump_gap_x;
Transfinite Surface {20006};

Curve Loop(10007) = {78, 1008, -178 , -1007};
Surface(20007) = {10007};
Transfinite Line {78}   = res * dx_width_channel Using Bump bump_square_y;
Transfinite Line {178}   = res * dx_width_channel Using Bump bump_square_y;
Transfinite Line {1008}  = res * dx_width_channel Using Bump bump_square_x;
Transfinite Surface {20007};


// second column
Curve Loop(10008) = {112, 1012, -212 , -1011};
Surface(20008) = {10008};
Transfinite Line {212}   = res_fine * dx_width_channel Using Bump bump_channel_y;
Transfinite Line {1012, 1011}  = res_coarse * l_channel Using Bump bump_channel_x;
Transfinite Surface {20008};

Curve Loop(10009) = {134, 1014, -234 , -1013};
Surface(20009) = {10009};
Transfinite Line {134,  234}   = res * dx_width_channel Using Bump bump_channel_y;
Transfinite Line {1014, 1013}  = res_coarse * l_channel Using Bump bump_channel_x;
Transfinite Surface {20009};

Curve Loop(10010) = {156, 1016, -256 , -1015};
Surface(20010) = {10010};
Transfinite Line {156,  256}   = res * dx_width_channel Using Bump bump_channel_y;
Transfinite Line {1016, 1015}  = res_coarse * l_channel Using Bump bump_channel_x;
Transfinite Surface {20010};

Curve Loop(10011) = {178, 1018, -278 , -1017};
Surface(20011) = {10011};
Transfinite Line {278}   = res * dx_width_channel Using Bump bump_channel_y;
Transfinite Line {1018, 1017}  = res_coarse * l_channel Using Bump bump_channel_x;
Transfinite Surface {20011};

// third column

Curve Loop(10012) = {212, 1022, -312 , -1021};
Surface(20012) = {10012};
Transfinite Line {312}   = res_fine * dx_width_channel Using Bump bump_square_y;
Transfinite Line {1021, 1022}  = res * dx_width_channel Using Bump bump_square_x;
Transfinite Surface {20012};

Curve Loop(10013) = {223, 1023, -323 , -1022};
Surface(20013) = {10013};
Transfinite Line {223,  323}   = res * dx_width_gap Using Bump bump_square_y;
Transfinite Line {1023}  = res * dx_width_channel Using Bump bump_square_x;
Transfinite Surface {20013};

Curve Loop(10014) = {234, 1024, -334 , -1023};
Surface(20014) = {10014};
Transfinite Line {234,  334}   = res * dx_width_channel Using Bump bump_square_y;
Transfinite Line {1024}  = res * dx_width_channel Using Bump bump_square_x;
Transfinite Surface {20014};

Curve Loop(10015) = {245, 1025, -345 , -1024};
Surface(20015) = {10015};
Transfinite Line {245,  345}   = res * dx_width_gap Using Bump bump_square_y;
Transfinite Line {1025}  = res * dx_width_channel Using Bump bump_square_x;
Transfinite Surface {20015};

Curve Loop(10016) = {256, 1026, -356 , -1025};
Surface(20016) = {10016};
Transfinite Line {256,  356}   = res * dx_width_channel Using Bump bump_square_y;
Transfinite Line {1026}  = res * dx_width_channel Using Bump bump_square_x;
Transfinite Surface {20016};

Curve Loop(10017) = {267, 1027, -367 , -1026};
Surface(20017) = {10017};
Transfinite Line {267,  367}   = res * dx_width_gap Using Bump bump_square_y;
Transfinite Line {1027}  = res * dx_width_channel Using Bump bump_square_x;
Transfinite Surface {20017};

Curve Loop(10018) = {278, 1028, -378 , -1027};
Surface(20018) = {10018};
Transfinite Line {278,  378}   = res * dx_width_channel Using Bump bump_square_y;
Transfinite Line {1028}  = res * dx_width_channel Using Bump bump_square_x;
Transfinite Surface {20018};

// Inlet channel

P9[] = {P1[0] -s_a * l_inflow , P1[1] -c_a * l_inflow};
P19[] = {P11[0] -s_a * l_inflow - s_a * d_off  , P11[1] -c_a * l_inflow - c_a * d_off};

Point(9) = {P9[0], P9[1], 0, scale};
Point(19) = {P19[0], P19[1], 0, scale};

Line(19) = {9, 1};
Line(119) = {19, 11};
//Horizontal
Line(1009) = {9, 19};

Curve Loop(10019) = {19, 1001, -119 , -1009};
Surface(20019) = {10019};
Transfinite Line {19,  119}   = res * l_inflow Using Bump bump_inlet_y;
Transfinite Line {1009}  = res * dx_width_channel Using Bump bump_inlet_x;
Transfinite Surface {20019};


Recombine Surface "*";

Physical Curve("inflow") = {1009};
Physical Curve("wall") = {19, 119, 12, 23, 123, 34, 45, 145, 56, 67, 167, 78, 312, 323, 334, 345, 356, 367, 378, 223, 245, 267, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1021, 1028, 1008};


