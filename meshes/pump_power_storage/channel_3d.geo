//l_side = 1.45;
dx_width_channel = 0.1;
dx_width_wall = 0.025;
dx_width_gap = 0.3;
dx_offset_channel = 2 * dx_width_wall + dx_width_gap;
l_inflow = 0.1;
dx_width_front = 0.1;
dx_width_back = 0.1;
l_channel = 2.09 - dx_width_front - dx_width_back;
alpha = 45;

h_water = 0.1;

// offset to merge the inflow to the outflow of the inlet
d_off = 0.1 * Sqrt(2)/2;

bump_base = 1/4;
bump_inlet_x = bump_base/1;
bump_inlet_y = bump_base/1;
bump_square_x = bump_base/1;
bump_square_y = bump_base/1;
bump_gap_x = bump_base/1;
bump_gap_y = bump_base/1;
bump_channel_x = bump_base/8;
bump_channel_y = bump_base/1;

bump_z = bump_base/1;

scale_z = 2.;
transfinite_scale = 5;
//transfinite_scale = 5;

s_a = Sin(alpha * Pi / 180);
c_a = Cos(alpha * Pi / 180);

height_water = 0.8;
height_air = 0.2;
N_layers = 30;
N_layers_water = N_layers * height_water/(height_air + height_water);
N_layers_air = N_layers * height_air/(height_air + height_water);

/* scale = 0.5; */
/* scale = 0.25; */
/* scale = 0.125; */
/* scale = 0.0625; */
scale = 0.03125;

res_coarse = transfinite_scale * 3.;
res = res_coarse * 10.;
res_fine = res * 1.0;

// First 
P1[] = {0, 0};
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
Point(13) = {P13[0], P13[1], 0, scale}; Point(14) = {P14[0], P14[1], 0, scale};
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

P9[] = {P1[0] -s_a * l_inflow , P1[1]-c_a * l_inflow};
P19[] = {P11[0] -s_a * l_inflow - s_a * d_off, P11[1] -c_a * l_inflow - c_a * d_off };

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

//---------------------------------------------------------
//-----------------------Layer 2---------------------------
//---------------------------------------------------------

// First 
P201[] = {0, 0};
P202[] = {P201[0] + s_a*dx_width_channel,  P201[1]+c_a*dx_width_channel};
P203[] = {P202[0] + s_a*dx_offset_channel, P202[1]+c_a*dx_offset_channel};
P204[] = {P203[0] + s_a*dx_width_channel,  P203[1]+c_a*dx_width_channel};
P205[] = {P204[0] + s_a*dx_offset_channel, P204[1]+c_a*dx_offset_channel};
P206[] = {P205[0] + s_a*dx_width_channel,  P205[1]+c_a*dx_width_channel};
P207[] = {P206[0] + s_a*dx_offset_channel, P206[1]+c_a*dx_offset_channel};
P208[] = {P207[0] + s_a*dx_width_channel,  P207[1]+c_a*dx_width_channel};

Point(201) = {P201[0], P201[1], h_water, scale};
Point(202) = {P202[0], P202[1], h_water, scale};
Point(203) = {P203[0], P203[1], h_water, scale};
Point(204) = {P204[0], P204[1], h_water, scale};
Point(205) = {P205[0], P205[1], h_water, scale};
Point(206) = {P206[0], P206[1], h_water, scale};
Point(207) = {P207[0], P207[1], h_water, scale};
Point(208) = {P208[0], P208[1], h_water, scale};

Line(2012) = {201, 202};
Line(2023) = {202, 203};
Line(2034) = {203, 204};
Line(2045) = {204, 205};
Line(2056) = {205, 206};
Line(2067) = {206, 207};
Line(2078) = {207, 208};

// Second
P2011[] = {P201[0] + dx_width_front, P201[1]};
P2012[] = {P202[0] + dx_width_front, P202[1]};
P2013[] = {P203[0] + dx_width_front, P203[1]};
P2014[] = {P204[0] + dx_width_front, P204[1]};
P2015[] = {P205[0] + dx_width_front, P205[1]};
P2016[] = {P206[0] + dx_width_front, P206[1]};
P2017[] = {P207[0] + dx_width_front, P207[1]};
P2018[] = {P208[0] + dx_width_front, P208[1]};

Point(2011) = {P2011[0], P2011[1], h_water, scale};
Point(2012) = {P2012[0], P2012[1], h_water, scale};
Point(2013) = {P2013[0], P2013[1], h_water, scale}; 
Point(2014) = {P2014[0], P2014[1], h_water, scale};
Point(2015) = {P2015[0], P2015[1], h_water, scale};
Point(2016) = {P2016[0], P2016[1], h_water, scale};
Point(2017) = {P2017[0], P2017[1], h_water, scale};
Point(2018) = {P2018[0], P2018[1], h_water, scale};

Line(20112) = {2011, 2012};
Line(20123) = {2012, 2013};
Line(20134) = {2013, 2014};
Line(20145) = {2014, 2015};
Line(20156) = {2015, 2016};
Line(20167) = {2016, 2017};
Line(20178) = {2017, 2018};

// Third
P2021[] = {P2011[0] + l_channel, P2011[1]};
P2022[] = {P2012[0] + l_channel, P2012[1]};
P2023[] = {P2013[0] + l_channel, P2013[1]};
P2024[] = {P2014[0] + l_channel, P2014[1]};
P2025[] = {P2015[0] + l_channel, P2015[1]};
P2026[] = {P2016[0] + l_channel, P2016[1]};
P2027[] = {P2017[0] + l_channel, P2017[1]};
P2028[] = {P2018[0] + l_channel, P2018[1]};

Point(2021) = {P2021[0], P2021[1], h_water, scale};
Point(2022) = {P2022[0], P2022[1], h_water, scale};
Point(2023) = {P2023[0], P2023[1], h_water, scale};
Point(2024) = {P2024[0], P2024[1], h_water, scale};
Point(2025) = {P2025[0], P2025[1], h_water, scale};
Point(2026) = {P2026[0], P2026[1], h_water, scale};
Point(2027) = {P2027[0], P2027[1], h_water, scale};
Point(2028) = {P2028[0], P2028[1], h_water, scale};

Line(20212) = {2021, 2022};
Line(20223) = {2022, 2023};
Line(20234) = {2023, 2024};
Line(20245) = {2024, 2025};
Line(20256) = {2025, 2026};
Line(20267) = {2026, 2027};
Line(20278) = {2027, 2028};

// Fourth
P2031[] = {P2021[0] + dx_width_back, P2021[1]};
P2032[] = {P2022[0] + dx_width_back, P2022[1]};
P2033[] = {P2023[0] + dx_width_back, P2023[1]};
P2034[] = {P2024[0] + dx_width_back, P2024[1]};
P2035[] = {P2025[0] + dx_width_back, P2025[1]};
P2036[] = {P2026[0] + dx_width_back, P2026[1]};
P2037[] = {P2027[0] + dx_width_back, P2027[1]};
P2038[] = {P2028[0] + dx_width_back, P2028[1]};

Point(2031) = {P2031[0], P2031[1], h_water, scale};
Point(2032) = {P2032[0], P2032[1], h_water, scale};
Point(2033) = {P2033[0], P2033[1], h_water, scale};
Point(2034) = {P2034[0], P2034[1], h_water, scale};
Point(2035) = {P2035[0], P2035[1], h_water, scale};
Point(2036) = {P2036[0], P2036[1], h_water, scale};
Point(2037) = {P2037[0], P2037[1], h_water, scale};
Point(2038) = {P2038[0], P2038[1], h_water, scale};

Line(20312) = {2031, 2032};
Line(20323) = {2032, 2033};
Line(20334) = {2033, 2034};
Line(20345) = {2034, 2035};
Line(20356) = {2035, 2036};
Line(20367) = {2036, 2037};
Line(20378) = {2037, 2038};

// Horizontals

Line(201001) = {201, 2011};
Line(201011) = {2011, 2021};
Line(201021) = {2021, 2031};

Line(201002) = {202, 2012};
Line(201012) = {2012, 2022};
Line(201022) = {2022, 2032};

Line(201003) = {203, 2013};
Line(201013) = {2013, 2023};
Line(201023) = {2023, 2033};

Line(201004) = {204, 2014};
Line(201014) = {2014, 2024};
Line(201024) = {2024, 2034};

Line(201005) = {205, 2015};
Line(201015) = {2015, 2025};
Line(201025) = {2025, 2035};

Line(201006) = {206, 2016};
Line(201016) = {2016, 2026};
Line(201026) = {2026, 2036};

Line(201007) = {207, 2017};
Line(201017) = {2017, 2027};
Line(201027) = {2027, 2037};

Line(201008) = {208, 2018};
Line(201018) = {2018, 2028};
Line(201028) = {2028, 2038};

// Curve loops
// Note: Transfinite Line takes the parallel lines as input

// first column
Curve Loop(2010001) = {2012, 201002, -20112 , -201001};
Surface(2020001) = {2010001};
Transfinite Line {2012}   = res_fine * dx_width_channel Using Bump bump_square_y;
Transfinite Line {20112}   = res_fine * dx_width_channel Using Bump bump_square_y;
Transfinite Line {201001, 201002}  = res * dx_width_channel Using Bump bump_square_x;
Transfinite Surface {2020001};

Curve Loop(2010002) = {2023, 201003, -20123 , -201002};
Surface(2020002) = {2010002};
Transfinite Line {2023}   = res * dx_width_gap Using Bump bump_gap_y;
Transfinite Line {20123}   = res * dx_width_gap Using Bump bump_gap_y;
Transfinite Line {201003}  = res * dx_width_channel Using Bump bump_gap_x;
Transfinite Surface {2020002};

Curve Loop(2010003) = {2034, 201004, -20134 , -201003};
Surface(2020003) = {2010003};
Transfinite Line {2034}   = res * dx_width_channel Using Bump bump_square_y;
Transfinite Line {20134}   = res * dx_width_channel Using Bump bump_square_y;
Transfinite Line {201004}  = res * dx_width_channel Using Bump bump_square_x;
Transfinite Surface {2020003};

Curve Loop(2010004) = {2045, 201005, -20145 , -201004};
Surface(2020004) = {2010004};
Transfinite Line {2045}   = res * dx_width_gap Using Bump bump_gap_y;
Transfinite Line {20145}   = res * dx_width_gap Using Bump bump_gap_y;
Transfinite Line {201005}  = res * dx_width_channel Using Bump bump_gap_x;
Transfinite Surface {2020004};

Curve Loop(2010005) = {2056, 201006, -20156 , -201005};
Surface(2020005) = {2010005};
Transfinite Line {2056}   = res * dx_width_channel Using Bump bump_square_y;
Transfinite Line {20156}   = res * dx_width_channel Using Bump bump_square_y;
Transfinite Line {201006}  = res * dx_width_channel Using Bump bump_square_x;
Transfinite Surface {2020005};

Curve Loop(2010006) = {2067, 201007, -20167 , -201006};
Surface(2020006) = {2010006};
Transfinite Line {2067}   = res * dx_width_gap Using Bump bump_gap_y;
Transfinite Line {20167}   = res * dx_width_gap Using Bump bump_gap_y;
Transfinite Line {201007}  = res * dx_width_channel Using Bump bump_gap_x;
Transfinite Surface {2020006};

Curve Loop(2010007) = {2078, 201008, -20178 , -201007};
Surface(2020007) = {2010007};
Transfinite Line {2078}   = res * dx_width_channel Using Bump bump_square_y;
Transfinite Line {20178}   = res * dx_width_channel Using Bump bump_square_y;
Transfinite Line {201008}  = res * dx_width_channel Using Bump bump_square_x;
Transfinite Surface {2020007};


// second column
Curve Loop(2010008) = {20112, 201012, -20212 , -201011};
Surface(2020008) = {2010008};
Transfinite Line {20212}   = res_fine * dx_width_channel Using Bump bump_channel_y;
Transfinite Line {201012, 201011}  = res_coarse * l_channel Using Bump bump_channel_x;
Transfinite Surface {2020008};

Curve Loop(2010009) = {20134, 201014, -20234 , -201013};
Surface(2020009) = {2010009};
Transfinite Line {20134,  20234}   = res * dx_width_channel Using Bump bump_channel_y;
Transfinite Line {201014, 201013}  = res_coarse * l_channel Using Bump bump_channel_x;
Transfinite Surface {2020009};

Curve Loop(2010010) = {20156, 201016, -20256 , -201015};
Surface(2020010) = {2010010};
Transfinite Line {20156,  20256}   = res * dx_width_channel Using Bump bump_channel_y;
Transfinite Line {201016, 201015}  = res_coarse * l_channel Using Bump bump_channel_x;
Transfinite Surface {2020010};

Curve Loop(2010011) = {20178, 201018, -20278 , -201017};
Surface(2020011) = {2010011};
Transfinite Line {20278}   = res * dx_width_channel Using Bump bump_channel_y;
Transfinite Line {201018, 201017}  = res_coarse * l_channel Using Bump bump_channel_x;
Transfinite Surface {2020011};

// third column

Curve Loop(2010012) = {20212, 201022, -20312 , -201021};
Surface(2020012) = {2010012};
Transfinite Line {20312}   = res_fine * dx_width_channel Using Bump bump_square_y;
Transfinite Line {201021, 201022}  = res * dx_width_channel Using Bump bump_square_x;
Transfinite Surface {2020012};

Curve Loop(2010013) = {20223, 201023, -20323 , -201022};
Surface(2020013) = {2010013};
Transfinite Line {20223,  20323}   = res * dx_width_gap Using Bump bump_square_y;
Transfinite Line {201023}  = res * dx_width_channel Using Bump bump_square_x;
Transfinite Surface {2020013};

Curve Loop(2010014) = {20234, 201024, -20334 , -201023};
Surface(2020014) = {2010014};
Transfinite Line {20234,  20334}   = res * dx_width_channel Using Bump bump_square_y;
Transfinite Line {201024}  = res * dx_width_channel Using Bump bump_square_x;
Transfinite Surface {2020014};

Curve Loop(2010015) = {20245, 201025, -20345 , -201024};
Surface(2020015) = {2010015};
Transfinite Line {20245,  20345}   = res * dx_width_gap Using Bump bump_square_y;
Transfinite Line {201025}  = res * dx_width_channel Using Bump bump_square_x;
Transfinite Surface {2020015};

Curve Loop(2010016) = {20256, 201026, -20356 , -201025};
Surface(2020016) = {2010016};
Transfinite Line {20256,  20356}   = res * dx_width_channel Using Bump bump_square_y;
Transfinite Line {201026}  = res * dx_width_channel Using Bump bump_square_x;
Transfinite Surface {2020016};

Curve Loop(2010017) = {20267, 201027, -20367 , -201026};
Surface(2020017) = {2010017};
Transfinite Line {20267,  20367}   = res * dx_width_gap Using Bump bump_square_y;
Transfinite Line {201027}  = res * dx_width_channel Using Bump bump_square_x;
Transfinite Surface {2020017};

Curve Loop(2010018) = {20278, 201028, -20378 , -201027};
Surface(2020018) = {2010018};
Transfinite Line {20278,  20378}   = res * dx_width_channel Using Bump bump_square_y;
Transfinite Line {201028}  = res * dx_width_channel Using Bump bump_square_x;
Transfinite Surface {2020018};

// Inlet channel

P209[] = {P201[0] -s_a * l_inflow, P201[1] -c_a * l_inflow};
P2019[] = {P2011[0] -s_a * l_inflow -s_a*d_off, P2011[1] -c_a * l_inflow - c_a*d_off};

Point(209) = {P209[0], P209[1], h_water, scale};
Point(2019) = {P2019[0], P2019[1], h_water, scale};

Line(2019) = {209, 201};
Line(20119) = {2019, 2011};
//Horizontal
Line(201009) = {209, 2019};

Curve Loop(2010019) = {2019, 201001, -20119 , -201009};
Surface(2020019) = {2010019};
Transfinite Line {2019,  20119}   = res * l_inflow Using Bump bump_inlet_y;
Transfinite Line {201009}  = res * dx_width_channel Using Bump bump_inlet_x;
Transfinite Surface {2020019};

//------------------------------------------------------
//----------------------Vertical lines------------------
//------------------------------------------------------


Line(3001) = {1, 201};
Line(3002) = {2, 202};
Line(3003) = {3, 203};
Line(3004) = {4, 204};
Line(3005) = {5, 205};
Line(3006) = {6, 206};
Line(3007) = {7, 207};
Line(3008) = {8, 208};

Line(3011) = {11, 2011};
Line(3012) = {12, 2012};
Line(3013) = {13, 2013};
Line(3014) = {14, 2014};
Line(3015) = {15, 2015};
Line(3016) = {16, 2016};
Line(3017) = {17, 2017};
Line(3018) = {18, 2018};

Line(3021) = {21, 2021};
Line(3022) = {22, 2022};
Line(3023) = {23, 2023};
Line(3024) = {24, 2024};
Line(3025) = {25, 2025};
Line(3026) = {26, 2026};
Line(3027) = {27, 2027};
Line(3028) = {28, 2028};

Line(3031) = {31, 2031};
Line(3032) = {32, 2032};
Line(3033) = {33, 2033};
Line(3034) = {34, 2034};
Line(3035) = {35, 2035};
Line(3036) = {36, 2036};
Line(3037) = {37, 2037};
Line(3038) = {38, 2038};

Line(3009) = {9, 209};
Line(3019) = {19, 2019};

//------------------------------------------------------
//----------------------Surfaces------------------------
//------------------------------------------------------

// Left, left side
Curve Loop(30019) = {19, 3001, -2019 , -3009};
Surface(40019) = {30019};
//Transfinite Line {3009}   = res_fine * dx_width_channel Using Bump bump_z;
Transfinite Line {3009}   = scale_z * res_fine * dx_width_channel Using Progression 1;
//Transfinite Line {3001}   = res_fine * dx_width_channel Using Bump bump_z;
Transfinite Line {3001}   = scale_z * res_fine * dx_width_channel Using Progression 1.;
Transfinite Surface {40019};

Curve Loop(30012) = {12, 3002, -2012 , -3001};
Surface(40012) = {30012};
//Transfinite Line {3002}   = res_fine * dx_width_channel Using Bump bump_z;
Transfinite Line {3002}   = scale_z * res_fine * dx_width_channel Using Progression 1.;
Transfinite Surface {40012};

Curve Loop(30023) = {23, 3003, -2023 , -3002};
Surface(40023) = {30023};
//Transfinite Line {3003}   = res_fine * dx_width_channel Using Bump bump_z;
Transfinite Line {3003}   = scale_z * res_fine * dx_width_channel Using Progression 1.;
Transfinite Surface {40023};

Curve Loop(30034) = {34, 3004, -2034 , -3003};
Surface(40034) = {30034};
//Transfinite Line {3004}   = res_fine * dx_width_channel Using Bump bump_z;
Transfinite Line {3004}   = scale_z * res_fine * dx_width_channel Using Progression 1.;
Transfinite Surface {40034};

Curve Loop(30045) = {45, 3005, -2045 , -3004};
Surface(40045) = {30045};
//Transfinite Line {3005}   = res_fine * dx_width_channel Using Bump bump_z;
Transfinite Line {3005}   = scale_z * res_fine * dx_width_channel Using Progression 1.;
Transfinite Surface {40045};

Curve Loop(30056) = {56, 3006, -2056 , -3005};
Surface(40056) = {30056};
//Transfinite Line {3006}   = res_fine * dx_width_channel Using Bump bump_z;
Transfinite Line {3006}   = scale_z * res_fine * dx_width_channel Using Progression 1.;
Transfinite Surface {40056};

Curve Loop(30067) = {67, 3007, -2067 , -3006};
Surface(40067) = {30067};
//Transfinite Line {3007}   = res_fine * dx_width_channel Using Bump bump_z;
Transfinite Line {3007}   = scale_z * res_fine * dx_width_channel Using Progression 1.;
Transfinite Surface {40067};

Curve Loop(30078) = {78, 3008, -2078 , -3007};
Surface(40078) = {30078};
//Transfinite Line {3008}   = res_fine * dx_width_channel Using Bump bump_z;
Transfinite Line {3008}   = scale_z * res_fine * dx_width_channel Using Progression 1.;
Transfinite Surface {40078};

// Left, right side
Curve Loop(30119) = {119, 3011, -20119 , -3019};
Surface(40119) = {-30119};
//Transfinite Line {3019}   = res_fine * dx_width_channel Using Bump bump_z;
Transfinite Line {3019}   = scale_z * res_fine * dx_width_channel Using Progression 1.;
//Transfinite Line {3011}   = res_fine * dx_width_channel Using Bump bump_z;
Transfinite Line {3011}   = scale_z * res_fine * dx_width_channel Using Progression 1.;
Transfinite Surface {40119};

Curve Loop(30112) = {112, 3012, -20112 , -3011};
Surface(40112) = {-30112};
//Transfinite Line {3012}   = res_fine * dx_width_channel Using Bump bump_z;
Transfinite Line {3012}   = scale_z * res_fine * dx_width_channel Using Progression 1.;
Transfinite Surface {40112};

Curve Loop(30123) = {123, 3013, -20123 , -3012};
Surface(40123) = {-30123};
//Transfinite Line {3013}   = res_fine * dx_width_channel Using Bump bump_z;
Transfinite Line {3013}   = scale_z * res_fine * dx_width_channel Using Progression 1.;
Transfinite Surface {40123};

Curve Loop(30134) = {134, 3014, -20134 , -3013};
Surface(40134) = {-30134};
//Transfinite Line {3014}   = res_fine * dx_width_channel Using Bump bump_z;
Transfinite Line {3014}   = scale_z * res_fine * dx_width_channel Using Progression 1.;
Transfinite Surface {40134};

Curve Loop(30145) = {145, 3015, -20145 , -3014};
Surface(40145) = {-30145};
//Transfinite Line {3015}   = res_fine * dx_width_channel Using Bump bump_z;
Transfinite Line {3015}   = scale_z * res_fine * dx_width_channel Using Progression 1.;
Transfinite Surface {40145};

Curve Loop(30156) = {156, 3016, -20156 , -3015};
Surface(40156) = {-30156};
//Transfinite Line {3016}   = res_fine * dx_width_channel Using Bump bump_z;
Transfinite Line {3016}   = scale_z * res_fine * dx_width_channel Using Progression 1.;
Transfinite Surface {40156};

Curve Loop(30167) = {167, 3017, -20167 , -3016};
Surface(40167) = {-30167};
//Transfinite Line {3017}   = res_fine * dx_width_channel Using Bump bump_z;
Transfinite Line {3017}   = scale_z * res_fine * dx_width_channel Using Progression 1.;
Transfinite Surface {40167};

Curve Loop(30178) = {178, 3018, -20178 , -3017};
Surface(40178) = {-30178};
//Transfinite Line {3018}   = res_fine * dx_width_channel Using Bump bump_z;
Transfinite Line {3018}   = scale_z * res_fine * dx_width_channel Using Progression 1.;
Transfinite Surface {40178};

// Left, lits

Curve Loop(31009) = {3009, 201009, -3019 , -1009};
Surface(41009) = {31009};
Transfinite Surface {41009};

Curve Loop(31001) = {3001, 201001, -3011 , -1001};
Surface(41001) = {31001};
Transfinite Surface {41001};

Curve Loop(31002) = {3002, 201002, -3012 , -1002};
Surface(41002) = {31002};
Transfinite Surface {41002};

Curve Loop(31003) = {3003, 201003, -3013 , -1003};
Surface(41003) = {31003};
Transfinite Surface {41003};

Curve Loop(31004) = {3004, 201004, -3014 , -1004};
Surface(41004) = {31004};
Transfinite Surface {41004};

Curve Loop(31005) = {3005, 201005, -3015 , -1005};
Surface(41005) = {31005};
Transfinite Surface {41005};

Curve Loop(31006) = {3006, 201006, -3016 , -1006};
Surface(41006) = {31006};
Transfinite Surface {41006};

Curve Loop(31007) = {3007, 201007, -3017 , -1007};
Surface(41007) = {31007};
Transfinite Surface {41007};

Curve Loop(31008) = {3008, 201008, -3018 , -1008};
Surface(41008) = {31008};
Transfinite Surface {41008};


//-------------------------------------------------------
//-----------------------Surfaces------------------------
//-------------------------------------------------------

// Left, mid

Curve Loop(31011) = {1011, 3021, -201011 , -3011};
//Transfinite Line {3021}   = res_fine * dx_width_channel Using Bump bump_z;
Transfinite Line {3021}   = scale_z * res_fine * dx_width_channel Using Progression 1.;
Surface(41011) = {31011};
Transfinite Surface {41011};

Curve Loop(31012) = {1012, 3022, -201012 , -3012};
//Transfinite Line {3022}   = res_fine * dx_width_channel Using Bump bump_z;
Transfinite Line {3022}   = scale_z * res_fine * dx_width_channel Using Progression 1.;
Surface(41012) = {31012};
Transfinite Surface {41012};

Curve Loop(31013) = {1013, 3023, -201013 , -3013};
//Transfinite Line {3023}   = res_fine * dx_width_channel Using Bump bump_z;
Transfinite Line {3023}   = scale_z * res_fine * dx_width_channel Using Progression 1.;
Surface(41013) = {31013};
Transfinite Surface {41013};

Curve Loop(31014) = {1014, 3024, -201014 , -3014};
//Transfinite Line {3024}   = res_fine * dx_width_channel Using Bump bump_z;
Transfinite Line {3024}   = scale_z * res_fine * dx_width_channel Using Progression 1.;
Surface(41014) = {31014};
Transfinite Surface {41014};

Curve Loop(31015) = {1015, 3025, -201015 , -3015};
//Transfinite Line {3025}   = res_fine * dx_width_channel Using Bump bump_z;
Transfinite Line {3025}   = scale_z * res_fine * dx_width_channel Using Progression 1.;
Surface(41015) = {31015};
Transfinite Surface {41015};

Curve Loop(31016) = {1016, 3026, -201016 , -3016};
//Transfinite Line {3026}   = res_fine * dx_width_channel Using Bump bump_z;
Transfinite Line {3026}   = scale_z * res_fine * dx_width_channel Using Progression 1.;
Surface(41016) = {31016};
Transfinite Surface {41016};

Curve Loop(31017) = {1017, 3027, -201017 , -3017};
//Transfinite Line {3027}   = res_fine * dx_width_channel Using Bump bump_z;
Transfinite Line {3027}   = scale_z * res_fine * dx_width_channel Using Progression 1.;
Surface(41017) = {31017};
Transfinite Surface {41017};

Curve Loop(31018) = {1018, 3028, -201018 , -3018};
//Transfinite Line {3028}   = res_fine * dx_width_channel Using Bump bump_z;
Transfinite Line {3028}   = scale_z * res_fine * dx_width_channel Using Progression 1.;
Surface(41018) = {31018};
Transfinite Surface {41018};

// Mid, right side

Curve Loop(30212) = {212, 3022, -20212 , -3021};
Surface(40212) = {30212};
Transfinite Surface {40212};

Curve Loop(30223) = {223, 3023, -20223 , -3022};
Surface(40223) = {30223};
Transfinite Surface {40223};

Curve Loop(30234) = {234, 3024, -20234 , -3023};
Surface(40234) = {30234};
Transfinite Surface {40234};

Curve Loop(30245) = {245, 3025, -20245 , -3024};
Surface(40245) = {30245};
Transfinite Surface {40245};

Curve Loop(30256) = {256, 3026, -20256 , -3025};
Surface(40256) = {30256};
Transfinite Surface {40256};

Curve Loop(30267) = {267, 3027, -20267 , -3026};
Surface(40267) = {30267};
Transfinite Surface {40267};

Curve Loop(30278) = {278, 3028, -20278 , -3027};
Surface(40278) = {30278};
Transfinite Surface {40278};

// Right, right side

Curve Loop(30312) = {312, 3032, -20312 , -3031};
Surface(40312) = {30312};
Transfinite Surface {40312};

Curve Loop(30323) = {323, 3033, -20323 , -3032};
Surface(40323) = {30323};
Transfinite Surface {40323};

Curve Loop(30334) = {334, 3034, -20334 , -3033};
Surface(40334) = {30334};
Transfinite Surface {40334};

Curve Loop(30345) = {345, 3035, -20345 , -3034};
Surface(40345) = {30345};
Transfinite Surface {40345};

Curve Loop(30356) = {356, 3036, -20356 , -3035};
Surface(40356) = {30356};
Transfinite Surface {40356};

Curve Loop(30367) = {367, 3037, -20367 , -3036};
Surface(40367) = {30367};
Transfinite Surface {40367};

Curve Loop(30378) = {378, 3038, -20378 , -3037};
Surface(40378) = {30378};
Transfinite Surface {40378};







//------------------------------------------------------------------------
//-------------------------Volumes mid------------------------------------
//------------------------------------------------------------------------

//Surface Loop(90009) = {20019, 2020019, 40019, 40119, 41009, 41001};
//Volume(99009) = {90009};
//Transfinite Volume {99009};
//
//Surface Loop(90001) = {20001, 2020001, 40012, 40112, 41002, 41001};
//Volume(99001) = {90001};
//Transfinite Volume {99001};
//
//Surface Loop(90002) = {20002, 2020002, 40023, 40123, 41003, 41002};
//Volume(99002) = {90002};
//Transfinite Volume {99002};
//
//Surface Loop(90003) = {20003, 2020003, 40034, 40134, 41004, 41003};
//Volume(99003) = {90003};
//Transfinite Volume {99003};
//
//Surface Loop(90004) = {20004, 2020004, 40045, 40145, 41005, 41004};
//Volume(99004) = {90004};
//Transfinite Volume {99004};
//
//Surface Loop(90005) = {20005, 2020005, 40056, 40156, 41006, 41005};
//Volume(99005) = {90005};
//Transfinite Volume {99005};
//
//Surface Loop(90006) = {20006, 2020006, 40067, 40167, 41007, 41006};
//Volume(99006) = {90006};
//Transfinite Volume {99006};
//
//Surface Loop(90007) = {20007, 2020007, 40078, 40178, 41008, 41007};
//Volume(99007) = {90007};
//Transfinite Volume {99007};

// Right, lits

Curve Loop(31021) = {3021, 201021, -3031 , -1021};
//Transfinite Line {3031}   = res_fine * dx_width_channel Using Bump bump_z;
Transfinite Line {3031}   = scale_z * res_fine * dx_width_channel Using Progression 1.;
Surface(41021) = {31021};
Transfinite Surface {41021};

Curve Loop(31022) = {3022, 201022, -3032 , -1022};
//Transfinite Line {3032}   = res_fine * dx_width_channel Using Bump bump_z;
Transfinite Line {3032}   = scale_z * res_fine * dx_width_channel Using Progression 1.;
Surface(41022) = {31022};
Transfinite Surface {41022};

Curve Loop(31023) = {3023, 201023, -3033 , -1023};
//Transfinite Line {3033}   = res_fine * dx_width_channel Using Bump bump_z;
Transfinite Line {3033}   = scale_z * res_fine * dx_width_channel Using Progression 1.;
Surface(41023) = {31023};
Transfinite Surface {41023};

Curve Loop(31024) = {3024, 201024, -3034 , -1024};
//Transfinite Line {3034}   = res_fine * dx_width_channel Using Bump bump_z;
Transfinite Line {3034}   = scale_z * res_fine * dx_width_channel Using Progression 1.;
Surface(41024) = {31024};
Transfinite Surface {41024};

Curve Loop(31025) = {3025, 201025, -3035 , -1025};
//Transfinite Line {3035}   = res_fine * dx_width_channel Using Bump bump_z;
Transfinite Line {3035}   = scale_z * res_fine * dx_width_channel Using Progression 1.;
Surface(41025) = {31025};
Transfinite Surface {41025};

Curve Loop(31026) = {3026, 201026, -3036 , -1026};
//Transfinite Line {3036}   = res_fine * dx_width_channel Using Bump bump_z;
Transfinite Line {3036}   = scale_z * res_fine * dx_width_channel Using Progression 1.;
Surface(41026) = {31026};
Transfinite Surface {41026};

Curve Loop(31027) = {3027, 201027, -3037 , -1027};
//Transfinite Line {3037}   = res_fine * dx_width_channel Using Bump bump_z;
Transfinite Line {3037}   = scale_z * res_fine * dx_width_channel Using Progression 1.;
//Transfinite Line {3038}   = res_fine * dx_width_channel Using Bump bump_z;
Transfinite Line {3038}   = scale_z * res_fine * dx_width_channel Using Progression 1.;
Surface(41027) = {31027};
Transfinite Surface {41027};

Curve Loop(31028) = {3028, 201028, -3038 , -1028};
Surface(41028) = {31028};
Transfinite Surface {41028};


//------------------------------------------------------------------------
//-------------------------Volumes left-----------------------------------
//------------------------------------------------------------------------

Surface Loop(90009) = {20019, 2020019, 40019, 40119, 41009, 41001};
Volume(99009) = {90009};
Transfinite Volume {99009};

Surface Loop(90001) = {20001, 2020001, 40012, 40112, 41002, 41001};
Volume(99001) = {90001};
Transfinite Volume {99001};

Surface Loop(90002) = {20002, 2020002, 40023, 40123, 41003, 41002};
Volume(99002) = {90002};
Transfinite Volume {99002};

Surface Loop(90003) = {20003, 2020003, 40034, 40134, 41004, 41003};
Volume(99003) = {90003};
Transfinite Volume {99003};

Surface Loop(90004) = {20004, 2020004, 40045, 40145, 41005, 41004};
Volume(99004) = {90004};
Transfinite Volume {99004};

Surface Loop(90005) = {20005, 2020005, 40056, 40156, 41006, 41005};
Volume(99005) = {90005};
Transfinite Volume {99005};

Surface Loop(90006) = {20006, 2020006, 40067, 40167, 41007, 41006};
Volume(99006) = {90006};
Transfinite Volume {99006};

Surface Loop(90007) = {20007, 2020007, 40078, 40178, 41008, 41007};
Volume(99007) = {90007};
Transfinite Volume {99007};

//------------------------------------------------------------------------
//-------------------------Volumes mid------------------------------------
//------------------------------------------------------------------------

Surface Loop(90011) = {20008, 2020008, 41011, 41012, 40112, 40212};
Volume(99011) = {90011};
Transfinite Volume {99011};

Surface Loop(90012) = {20009, 2020009, 41013, 41014, 40134, 40234};
Volume(99012) = {90012};
Transfinite Volume {99012};

Surface Loop(90013) = {20010, 2020010, 41015, 41016, 40156, 40256};
Volume(99013) = {90013};
Transfinite Volume {99013};

Surface Loop(90014) = {20011, 2020011, 41017, 41018, 40178, 40278};
Volume(99014) = {90014};
Transfinite Volume {99014};

//------------------------------------------------------------------------
//-------------------------Volumes right------------------------------------
//------------------------------------------------------------------------

Surface Loop(91012) = {20012, 2020012, 41021, 41022, 40212, 40312};
Volume(98012) = {91012};
Transfinite Volume {98012};

Surface Loop(91013) = {20013, 2020013, 41022, 41023, 40223, 40323};
Volume(98013) = {91013};
Transfinite Volume {98013};

Surface Loop(91014) = {20014, 2020014, 41023, 41024, 40234, 40334};
Volume(98014) = {91014};
Transfinite Volume {98014};

Surface Loop(91015) = {20015, 2020015, 41024, 41025, 40245, 40345};
Volume(98015) = {91015};
Transfinite Volume {98015};

Surface Loop(91016) = {20016, 2020016, 41025, 41026, 40256, 40356};
Volume(98016) = {91016};
Transfinite Volume {98016};

Surface Loop(91017) = {20017, 2020017, 41026, 41027, 40267, 40367};
Volume(98017) = {91017};
Transfinite Volume {98017};

Surface Loop(91018) = {20018, 2020018, 41027, 41028, 40278, 40378};
Volume(98018) = {91018};
Transfinite Volume {98018};


Recombine Surface "*";

