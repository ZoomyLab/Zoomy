l_1 = 0.05;
a_1 = 30.*3.14/180;
l_2 = 0.12;
a_2 = 20.*3.14/180;
l_inflow = 0.5;
l_nozzle_inflow_interpection = 0.3;

h_water = 0.10;
//h_air = 0.08;


h_inflow = 0.03;

//width = 0.05 *0.707;
width = 0.05 *Sqrt(2)/2;
z_in = 0.02;

transfinite_scaling = 5;
//transfinite_scaling = 5;
//transfinite_scaling = 10;
//transfinite_scaling = 15;

//transfinite_scaling = 5;


scale = 0.0125;

// Scaling for transfinite meshes
dx_in = 1.;
dy_in = 1.;
dz_in = 1.;
dz_in = 1.;
dx_diffusor = 1.;
dy_diffusor = 1.;
dz_diffusor = 1.;
dx_connector = 1.;
dy_connector = 1.;
dz_connector = 1.;
dx_channel_p1 = 1.;
dy_channel_p1 = 1.;
dz_channel_p1_w = 1.;
dz_channel_p1_a = 1.;
dx_channel_p2 = 1.;
dy_channel_p2 = 1.;
dz_channel_p2_w = 1.;
dz_channel_p2_a = 1.;

trans_dx_in = transfinite_scaling *  3;
trans_dx_diffusor = transfinite_scaling *  6;
trans_dx_connector = transfinite_scaling *  10;
trans_dx_channel_p1= trans_dx_connector ;
trans_dx_channel_p2 = transfinite_scaling *  3;

trans_dy_in = transfinite_scaling *  3;
trans_dy_diffusor =  trans_dy_in;
trans_dy_connector =  trans_dy_in;
trans_dy_channel_p1=  trans_dy_in;
trans_dy_channel_p2 =  trans_dy_in;

scale_bump = 1/4;

trans_dz_in = transfinite_scaling *  3;
trans_dz_diffusor =  trans_dz_in;
trans_dz_connector =  trans_dz_in;
trans_dz_channel_p1_w= transfinite_scaling *  6;
trans_dz_channel_p2_w = transfinite_scaling *  6;

//offset = 0;
offset = 1230000;


/* water block */
Point(offset + 1) = {0, 0, -width, scale};
Point(offset + 2) = {0,h_water, -width, scale};
Point(offset + 3) = {l_nozzle_inflow_interpection, h_water, -width, scale};
Point(offset + 4) = {l_nozzle_inflow_interpection, 0, -width, scale};
Point(offset + 5) = {l_inflow, 0, -width, scale};
Point(offset + 6) = {l_inflow, h_water, -width, scale};

Point(offset + 101) = {0, 0, width, scale};
Point(offset + 102) = {0,h_water, width, scale};
Point(offset + 103) = {l_nozzle_inflow_interpection, h_water, width, scale};
Point(offset + 104) = {l_nozzle_inflow_interpection, 0, width, scale};
Point(offset + 105) = {l_inflow, 0, width, scale};
Point(offset + 106) = {l_inflow, h_water, width, scale};

/* air block */
//Point(offset + 11) = {0, h_water+h_air, -width, scale};
//Point(offset + 12) = {l_nozzle_inflow_interpection, h_water+h_air, -width, scale};
//Point(offset + 13) = {l_inflow, h_water+h_air, -width, scale};
//
//Point(offset + 111) = {0, h_water+h_air, width, scale};
//Point(offset + 112) = {l_nozzle_inflow_interpection, h_water+h_air, width, scale};
//Point(offset + 113) = {l_inflow, h_water+h_air, width, scale};

/* nozzle left*/
Point(offset + 21) = {-l_2 * Cos(a_1 + a_2),- l_2 * Sin(a_1+a_2), -z_in, scale};
Point(offset + 22) = {-l_2 * Cos(a_1 + a_2) - l_1 * Cos(a_1) ,- l_2 * Sin(a_1+a_2) - l_1 * Sin(a_1), -z_in, scale};

Point(offset + 121) = {-l_2 * Cos(a_1 + a_2),- l_2 * Sin(a_1+a_2), z_in, scale};
Point(offset + 122) = {-l_2 * Cos(a_1 + a_2) - l_1 * Cos(a_1) ,- l_2 * Sin(a_1+a_2) - l_1 * Sin(a_1), z_in, scale};


/* nozzle right*/
Point(offset + 31) = {-l_2 * Cos(a_1 + a_2) - l_1 * Cos(a_1) + Sin(a_1) * h_inflow , - l_2 * Sin(a_1+a_2) - l_1 * Sin(a_1) - Cos(a_1)*h_inflow, -z_in, scale};
Point(offset + 32) = {-l_2 * Cos(a_1+a_2) - l_1 * Cos(a_1) + Sin(a_1) * h_inflow + l_1 * Cos(a_1) , - l_2 * Sin(a_1+a_2) - l_1 * Sin(a_1) - Cos(a_1)*h_inflow + l_1 * Sin(a_1) , -z_in, scale};
Point(offset + 33) = {-l_2 * Cos(a_1 + a_2) - l_1 * Cos(a_1) + Sin(a_1) * h_inflow + l_1 * Cos(a_1)  + l_2 * Cos(-a_2+a_1), - l_2 * Sin(a_1+a_2) - l_1 * Sin(a_1) - Cos(a_1)*h_inflow + l_1 * Sin(a_1) + l_2 * Sin(-a_2+a_1), -z_in, scale};

Point(offset + 131) = {-l_2 * Cos(a_1 + a_2) - l_1 * Cos(a_1) + Sin(a_1) * h_inflow , - l_2 * Sin(a_1+a_2) - l_1 * Sin(a_1) - Cos(a_1)*h_inflow, z_in, scale};
Point(offset + 132) = {-l_2 * Cos(a_1+a_2) - l_1 * Cos(a_1) + Sin(a_1) * h_inflow + l_1 * Cos(a_1) , - l_2 * Sin(a_1+a_2) - l_1 * Sin(a_1) - Cos(a_1)*h_inflow + l_1 * Sin(a_1) , z_in, scale};
Point(offset + 133) = {-l_2 * Cos(a_1 + a_2) - l_1 * Cos(a_1) + Sin(a_1) * h_inflow + l_1 * Cos(a_1)  + l_2 * Cos(-a_2+a_1), - l_2 * Sin(a_1+a_2) - l_1 * Sin(a_1) - Cos(a_1)*h_inflow + l_1 * Sin(a_1) + l_2 * Sin(-a_2+a_1), z_in, scale};

  
/* air block*/
Line(offset + 51) = {offset + 2,offset +  3};
//Line(offset + 52) = {offset + 3, offset + 12};
//Line(offset + 53) = {offset + 12,offset + 11};
//Line(offset + 54) = {offset + 11,offset + 2};
Line(offset + 55) = {offset + 3, offset +  6};
//Line(offset + 56) = {offset + 6, offset +  13};
//Line(offset + 57) = {offset + 13, offset +  12};

// p2 wall
//Transfinite Line {offset + 56} = trans_dz_channel_p2_a Using Progression dz_channel_p2_a;
Transfinite Line {offset + 57}  = trans_dx_channel_p2 Using Progression dx_channel_p2;
Transfinite Line {offset + 55} = trans_dx_channel_p2 Using Progression dx_channel_p2;
Transfinite Surface { 501011 };



Line(offset + 151) = {offset + 102, offset + 103};
//Line(offset + 152) = {offset + 103, offset + 112};
//Line(offset + 153) = {offset + 112, offset + 111};
//Line(offset + 154) = {offset + 111, offset + 102};

Line(offset + 155) = {offset + 103, offset + 106};
//Line(offset + 156) = {offset + 106, offset + 113};
//Line(offset + 157) = {offset + 113, offset + 112};



/* water block*/
Line(offset + 61) = {offset + 2, offset + 3};
Line(offset + 62) = {offset + 3, offset + 4};
Line(offset + 63) = {offset + 4, offset + 1};
Line(offset + 64) = {offset + 1, offset + 2};
Line(offset + 65) = {offset + 4, offset + 5};
Line(offset + 66) = {offset + 5, offset + 6};

// wapp p2
Transfinite Line {offset + 66} = trans_dz_channel_p2_w Using Progression dz_channel_p2_w;
Transfinite Line {offset + 55}  = trans_dx_channel_p2 Using Progression dx_channel_p2;
Transfinite Line {offset + 65} = trans_dx_channel_p2 Using Progression dx_channel_p2;
Transfinite Surface { offset + 501004 };

Line(offset + 161) = {offset + 102, offset + 103};
Line(offset + 162) = {offset + 103, offset + 104};
Line(offset + 163) = {offset + 104, offset + 101};
Line(offset + 164) = {offset + 101, offset + 102};
Line(offset + 165) = {offset + 104, offset + 105};
Line(offset + 166) = {offset + 105, offset + 106};


// wall p2
Transfinite Line {offset + 166} = trans_dz_channel_p2_w Using Progression dz_channel_p2_w;
Transfinite Line {offset + 155}  = trans_dx_channel_p2 Using Progression dx_channel_p2;
Transfinite Line {offset + 165} = trans_dx_channel_p2 Using Progression dx_channel_p2;
Transfinite Surface { offset + 501104 };


/* water nozzle top*/
Line(offset + 71) = {offset + 1, offset + 33};
Line(offset + 72) = {offset + 33, offset + 4};

Line(offset + 171) = {offset + 101, offset + 133};
Line(offset + 172) = {offset + 133, offset + 104};


/* water nozzle mid*/
Line(offset + 81) = {offset + 1, offset + 21};
Line(offset + 82) = {offset + 21,offset +  32};
Line(offset + 83) = {offset + 32,offset +  33};

Line(offset + 181) = {offset + 101, offset + 121};
Line(offset + 182) = {offset + 121, offset + 132};
Line(offset + 183) = {offset + 132, offset + 133};

/* water nozzle inflow */
Line(offset + 91) = {offset + 21, offset + 22};
Line(offset + 92) = {offset + 22, offset + 31};
Line(offset + 93) = {offset + 31, offset + 32};

Line(offset + 191) = {offset + 121, offset + 122};
Line(offset + 192) = {offset + 122, offset + 131};
Line(offset + 193) = {offset + 131, offset + 132};


/* z-direction */

Line(offset + 201) = {offset + 1, offset + 101};
Line(offset + 202) = {offset + 2, offset + 102};
Line(offset + 203) = {offset + 3, offset + 103};
Line(offset + 204) = {offset + 4, offset + 104};
Line(offset + 205) = {offset + 5, offset + 105};
Line(offset + 206) = {offset + 6, offset + 106};

//Line(offset + 211) = {offset + 11, offset + 111};
//Line(offset + 212) = {offset + 12, offset + 112};
//Line(offset + 213) = {offset + 13, offset + 113};

Line(offset + 221) = {offset + 21, offset + 121};
Line(offset + 222) = {offset + 22, offset + 122};

Line(offset + 231) = {offset + 31, offset + 131};
Line(offset + 232) = {offset + 32, offset + 132};
Line(offset + 233) = {offset + 33, offset + 133};

// inflow

Curve Loop(offset + 62000) = {-(offset+92), offset+222, offset+192, -(offset+231)};
Surface(offset +72000) = {offset+62000};

//Transfinite Line {offset+231}  = trans_dy_in Using Progression dy_in;
Transfinite Line {offset+231}  = trans_dy_in Using Bump scale_bump;
//Transfinite Line {offset+222}  = trans_dy_in Using Progression dy_in;
Transfinite Line {offset+222}  = trans_dy_in Using Bump scale_bump;
Transfinite Line {offset+192}  = trans_dz_in Using Progression dz_in;
Transfinite Line {offset+92}   = trans_dz_in Using Progression dz_in;
Transfinite Surface { offset+72000 };

// inflow walls

Curve Loop (offset+62001) = {offset+91, offset+92, offset+93, -(offset+82)};
Surface (offset+72001) = {offset+62001};
Transfinite Line {offset+92}  = trans_dz_in Using Progression dz_in;
Transfinite Line {offset+82}  = trans_dz_in Using Progression dz_in;
Transfinite Line {offset+93}  = trans_dx_in Using Progression dx_in;
Transfinite Line {offset+91}  = trans_dx_in Using Progression dx_in;
Transfinite Surface { offset+72001 };

Curve Loop (offset+62002) = {-(offset+191), offset+182, -(offset+193), -(offset+192)};
Surface (offset+72002) = {offset+62002};
Transfinite Line {offset+192}  = trans_dz_in Using Progression dz_in;
Transfinite Line {offset+182}  = trans_dz_in Using Progression dz_in;
Transfinite Line {offset+193}  = trans_dx_in Using Progression dx_in;
Transfinite Line {offset+191}  = trans_dx_in Using Progression dx_in;
Transfinite Surface { offset+72002 };

Curve Loop (offset+62003) = {offset+93, offset+232, -(offset+193), -(offset+231)};
Surface (offset+72003) = {offset+62003};
//Transfinite Line {offset+231}  = trans_dy_in Using Progression dy_in;
Transfinite Line {offset+231}  = trans_dy_in Using Bump scale_bump;
//Transfinite Line {offset+232}  = trans_dy_in Using Progression dy_in;
Transfinite Line {offset+232}  = trans_dy_in Using Bump scale_bump;
Transfinite Line {offset+93}  = trans_dx_in Using Progression dx_in;
Transfinite Line {offset+193}  = trans_dx_in Using Progression dx_in;
Transfinite Surface { offset+72003 };

Curve Loop (offset+62004) = {-(offset+91), offset+221, offset+191, -(offset+222)};
Surface (offset+72004) = {offset+62004};
//Transfinite Line {offset+222}  = trans_dy_in Using Progression dy_in;
Transfinite Line {offset+222}  = trans_dy_in Using Bump scale_bump;
//Transfinite Line {offset+221}  = trans_dy_in Using Progression dy_in;
Transfinite Line {offset+221}  = trans_dy_in Using Bump scale_bump;
Transfinite Line {offset+91}  = trans_dx_in Using Progression dx_in;
Transfinite Line {offset+191}  = trans_dx_in Using Progression dx_in;
Transfinite Surface { offset+72004 };

// Diffusor
Curve Loop (offset+62005) = {offset+82, offset+83, -(offset+71), offset+81};
Surface (offset+72005) = {offset+62005};
Transfinite Line {offset+82}  = trans_dz_diffusor Using Progression dz_diffusor;
Transfinite Line {offset+71}  = trans_dz_diffusor Using Progression dz_diffusor;
Transfinite Line {offset+83}  = trans_dx_diffusor Using Progression dx_diffusor;
Transfinite Line {offset+81}  = trans_dx_diffusor Using Progression dx_diffusor;
Transfinite Surface { offset+72005 };

Curve Loop (offset+62006) = {offset+182, offset+183, -(offset+171), offset+181};
Surface (offset+72006) = {offset+62006};
Transfinite Line {offset+182}  = trans_dz_diffusor Using Progression dz_diffusor;
Transfinite Line {offset+171}  = trans_dz_diffusor Using Progression dz_diffusor;
Transfinite Line {offset+183}  = trans_dx_diffusor Using Progression dx_diffusor;
Transfinite Line {offset+181}  = trans_dx_diffusor Using Progression dx_diffusor;
Transfinite Surface { offset+72006 };

Curve Loop (offset+62007) = {offset+83, offset+233, -(offset+183), -(offset+232)};
Surface (offset+72007) = {offset+62007};
//Transfinite Line {offset+233}  = trans_dy_diffusor Using Progression dy_diffusor;
Transfinite Line {offset+233}  = trans_dy_diffusor Using Bump scale_bump;
//Transfinite Line {offset+232}  = trans_dy_diffusor Using Progression dy_diffusor;
Transfinite Line {offset+232}  = trans_dy_diffusor Using Bump scale_bump;
Transfinite Line {offset+183}  = trans_dx_diffusor Using Progression dx_diffusor;
Transfinite Line {offset+83}   = trans_dx_diffusor Using Progression dx_diffusor;
Transfinite Surface { offset+72007 };

Curve Loop (offset+62008) = {offset+201, offset+181, -(offset+221), -(offset+81)};
Surface (offset+72008) = {offset+62008};
//Transfinite Line {offset+201}  = trans_dy_diffusor Using Progression dy_diffusor;
Transfinite Line {offset+201}  = trans_dy_diffusor Using Bump scale_bump;
//Transfinite Line {offset+221}  = trans_dy_diffusor Using Progression dy_diffusor;
Transfinite Line {offset+221}  = trans_dy_diffusor Using Bump scale_bump;
Transfinite Line {offset+181}  = trans_dx_diffusor Using Progression dx_diffusor;
Transfinite Line {offset+81}   = trans_dx_diffusor Using Progression dx_diffusor;
Transfinite Surface { offset+72008 };

// Diffusor into channel

Curve Loop (offset+62009) = {offset+163, offset+172, offset+171};
Surface (offset+72009) = {offset+62009};
Transfinite Line {offset+163}  = trans_dx_connector Using Progression dx_connector;
Transfinite Line {offset+172}   = trans_dx_connector Using Progression dx_connector;
Transfinite Surface { offset+72009 };

Curve Loop (offset+62010) = {offset+63, offset+72, offset+71};
Surface (offset+72010) = {offset+62010};
Transfinite Line {offset+63}  = trans_dx_connector Using Progression dx_connector;
Transfinite Line {offset+72}   = trans_dx_connector Using Progression dx_connector;
Transfinite Surface { offset+72010 };

// P1
Curve Loop (offset+62011) = {-(offset+51), -(offset+64), -(offset+63), -(offset+62)};
Surface (offset+72011) = {offset+62011};
Transfinite Line {offset+64}  = trans_dz_channel_p1_w Using Progression dz_channel_p1_w;
Transfinite Line {offset+62}  = trans_dz_channel_p1_w Using Progression dz_channel_p1_w;
Transfinite Line {offset+51}  = trans_dx_channel_p1 Using Progression dx_channel_p1;
Transfinite Line {offset+63}   = trans_dx_channel_p1 Using Progression dx_channel_p1;
Transfinite Surface { offset+72011 };

Curve Loop (offset+62012) = {-(offset+151), -(offset+164), -(offset+163), -(offset+162)};
Surface (offset+72012) = {offset+62012};
Transfinite Line {offset+164}  = trans_dz_channel_p1_w Using Progression dz_channel_p1_w;
Transfinite Line {offset+162}  = trans_dz_channel_p1_w Using Progression dz_channel_p1_w;
Transfinite Line {offset+151}  = trans_dx_channel_p1 Using Progression dx_channel_p1;
Transfinite Line {offset+163}   = trans_dx_channel_p1 Using Progression dx_channel_p1;
Transfinite Surface { offset + 72012 };

//Curve Loop (offset+62013) = {offset+51, offset+52, offset+53, offset+54};
//Surface (offset+72013) = {offset+62013};
//Transfinite Line {offset+52}  = trans_dz_channel_p1_a Using Progression dz_channel_p1_a;
//Transfinite Line {offset+54}  = trans_dz_channel_p1_a Using Progression dz_channel_p1_a;
//Transfinite Line {offset+51}  = trans_dx_channel_p1 Using Progression dx_channel_p1;
//Transfinite Line {offset+53}   = trans_dx_channel_p1 Using Progression dx_channel_p1;
//Transfinite Surface { offset+72013 };

//Curve Loop (offset+62014) = {offset+151, offset+152, offset+153, offset+154};
//Surface (offset+72014) = {offset+62014};
//Transfinite Line {offset+152}  = trans_dz_channel_p1_a Using Progression dz_channel_p1_a;
//Transfinite Line {offset+154}  = trans_dz_channel_p1_a Using Progression dz_channel_p1_a;
//Transfinite Line {offset+151}  = trans_dx_channel_p1 Using Progression dx_channel_p1;
//Transfinite Line {offset+153}   = trans_dx_channel_p1 Using Progression dx_channel_p1;
//Transfinite Surface { offset+72014 };

Curve Loop (offset+62015) = {offset+201, offset+164, -(offset+202), -(offset+64)};
Surface (offset+72015) = {offset+62015};
Transfinite Line {offset+164}  = trans_dz_channel_p1_w Using Progression dz_channel_p1_w;
Transfinite Line {offset+64}  = trans_dz_channel_p1_w Using Progression dz_channel_p1_w;
//Transfinite Line {offset+201}  = trans_dy_channel_p1 Using Progression dy_channel_p1;
Transfinite Line {offset+201}  = trans_dy_channel_p1 Using Bump scale_bump;
//Transfinite Line {offset+202}   = trans_dy_channel_p1 Using Progression dy_channel_p1;
Transfinite Line {offset+202}   = trans_dy_channel_p1 Using Bump scale_bump;
Transfinite Surface { offset+72015 };

//Curve Loop (offset+62016) = {offset+202, -(offset+154), -(offset+211), offset+54};
//Surface (offset+72016) = {offset+62016};
//Transfinite Line {offset+154}  = trans_dz_channel_p1_a Using Progression dz_channel_p1_a;
//Transfinite Line {offset+54}  = trans_dz_channel_p1_a Using Progression dz_channel_p1_a;
////Transfinite Line {offset+202}  = trans_dy_channel_p1 Using Progression dy_channel_p1;
//Transfinite Line {offset+202}  = trans_dy_channel_p1 Bump scale_bump;
////Transfinite Line {offset+211}   = trans_dy_channel_p1 Using Progression dy_channel_p1;
//Transfinite Line {offset+211}   = trans_dy_channel_p1 Bump scale_bump;
//Transfinite Surface { offset+72016 };

//Curve Loop (offset+62017) = {offset+53, offset+211, -(offset+153), -(offset+212)};
//Surface (offset+72017) = {offset+62017};
//Transfinite Line {offset+53}   = trans_dx_channel_p1 Using Progression dx_channel_p1;
//Transfinite Line {offset+153}  = trans_dx_channel_p1 Using Progression dx_channel_p1;
////Transfinite Line {offset+211}  = trans_dy_channel_p1 Using Progression dy_channel_p1;
//Transfinite Line {offset+211}  = trans_dy_channel_p1 Bump scale_bump;
////Transfinite Line {offset+212}  = trans_dy_channel_p1 Using Progression dy_channel_p1;
//Transfinite Line {offset+212}  = trans_dy_channel_p1 Bump scale_bump;
//Transfinite Surface { offset+72017 };

//connector bottom
Curve Loop (offset+62099) = {offset+233, offset+172, -(offset+204), -(offset+72 )};
Surface (offset+72099) = {offset+62099};
Transfinite Line {offset+72}   = trans_dx_connector Using Progression dx_connector;
Transfinite Line {offset+172}  = trans_dx_connector Using Progression dx_connector;
//Transfinite Line {offset+233}  = trans_dy_connector Using Progression dy_connector;
Transfinite Line {offset+233}  = trans_dy_connector Using Bump scale_bump;
//Transfinite Line {offset+204}  = trans_dy_connector Using Progression dy_connector;
Transfinite Line {offset+204}  = trans_dy_connector Using Bump scale_bump;
Transfinite Surface { offset+72099 };

// Channel


Curve Loop (offset+620018) = {offset+162, offset+165, offset+166, -(offset+155)};
Surface (offset+720018) = {offset+620018};

Transfinite Line {offset+162}  = trans_dz_channel_p2_w Using Progression dz_channel_p2_w;
Transfinite Line {offset+166} = trans_dz_channel_p2_w Using Progression dz_channel_p2_w;
Transfinite Line {offset+155}  = trans_dx_channel_p2 Using Progression dx_channel_p2;
Transfinite Line {offset+165} = trans_dx_channel_p2 Using Progression dx_channel_p2;
Transfinite Surface { offset+720018 };

Curve Loop (offset+620019) = {offset+62, offset+65, offset+66, -(offset+55)};
Surface (offset+720019) = {offset+620019};

Transfinite Line {offset+55}  = trans_dx_channel_p2 Using Progression dx_channel_p2;
Transfinite Line {offset+65} = trans_dx_channel_p2 Using Progression dx_channel_p2;
Transfinite Line {offset+62}  = trans_dz_channel_p2_w Using Progression dz_channel_p2_w;
Transfinite Line {offset+66} = trans_dz_channel_p2_w Using Progression dz_channel_p2_w;
Transfinite Surface { offset+720019 };

//Curve Loop (offset+620020) = {-(offset+152), offset+155, offset+156, offset+157};
//Surface (offset+720020) = {offset+620020};
//Transfinite Line {offset+152}  = trans_dz_channel_p2_a Using Progression dz_channel_p2_a;
//Transfinite Line {offset+156} = trans_dz_channel_p2_a Using Progression dz_channel_p2_a;
//Transfinite Line {offset+155}  = trans_dx_channel_p2 Using Progression dx_channel_p2;
//Transfinite Line {offset+157} = trans_dx_channel_p2 Using Progression dx_channel_p2;
//Transfinite Surface { offset+720020 };

//Curve Loop (offset+620021) = {-(offset+52), offset+55, offset+56, offset+57};
//Surface (offset+720021) = {offset+620021};
//Transfinite Line {offset+52}  = trans_dz_channel_p1_a Using Progression dz_channel_p1_a;
//Transfinite Line {offset+54} = trans_dz_channel_p1_a Using Progression dz_channel_p1_a;
//Transfinite Line {offset+53}  = trans_dx_channel_p1 Using Progression dx_channel_p1;
//Transfinite Line {offset+51} = trans_dx_channel_p1 Using Progression dx_channel_p1;
//Transfinite Surface { offset+720021 };

//bottom p2
Curve Loop (offset+620022) = {-(offset+65), offset+204, offset+165, -(offset+205)};
Surface (offset+720022) = {offset+620022};
//Transfinite Line {offset+204}  = trans_dy_channel_p2 Using Progression dy_channel_p2;
Transfinite Line {offset+204}  = trans_dy_channel_p2 Using Bump scale_bump;
//Transfinite Line {offset+205} = trans_dy_channel_p2 Using Progression dy_channel_p2;
Transfinite Line {offset+205} = trans_dy_channel_p2 Using Bump scale_bump;
Transfinite Line {offset+65}  = trans_dx_channel_p2 Using Progression dx_channel_p2;
Transfinite Line {offset+165} = trans_dx_channel_p2 Using Progression dx_channel_p2;
Transfinite Surface { offset+720022 };

// top p2
//Curve Loop (offset+620023) = {offset+57, offset+212, -(offset+157), -(offset+213)};
//Surface (offset+720023) = {offset+620023};
//Transfinite Line {offset+57}  = trans_dx_channel_p2 Using Progression dx_channel_p2;
//Transfinite Line {offset+157} = trans_dx_channel_p2 Using Progression dx_channel_p2;
////Transfinite Line {offset+212}  = trans_dy_channel_p2 Using Progression dy_channel_p2;
//Transfinite Line {offset+212}  = trans_dy_channel_p2 Bump scale_bump;
////Transfinite Line {offset+213} = trans_dy_channel_p2 Using Progression dy_channel_p2;
//Transfinite Line {offset+213} = trans_dy_channel_p2 Bump scale_bump;
//Transfinite Surface { offset+720023 };

//outflow water
Curve Loop (offset+620024) = {offset+205, offset+166, -(offset+206), -(offset+66)};
Surface (offset+720024) = {offset+620024};
Transfinite Line {offset+166}  = trans_dz_channel_p2_w Using Progression dz_channel_p2_w;
Transfinite Line {offset+66} = trans_dz_channel_p2_w Using Progression dz_channel_p2_w;
//Transfinite Line {offset+205}  = trans_dy_channel_p2 Using Progression dy_channel_p2;
Transfinite Line {offset+205}  = trans_dy_channel_p2 Using Bump scale_bump;
//Transfinite Line {offset+206} = trans_dy_channel_p2 Using Progression dy_channel_p2;
Transfinite Line {offset+206} = trans_dy_channel_p2 Using Bump scale_bump;
Transfinite Surface { offset+720024 };

//Curve Loop (offset+620025) = {offset+206, offset+156, -(offset+213), -(offset+56)};
//Surface (offset+720025) = {offset+620025};
// outflow air
//Transfinite Line {offset+56}  = trans_dz_channel_p2_a Using Progression dz_channel_p2_a;
//Transfinite Line {offset+156} = trans_dz_channel_p2_a Using Progression dz_channel_p2_a;
////Transfinite Line {offset+206}  = trans_dy_channel_p2 Using Progression dy_channel_p2;
//Transfinite Line {offset+206}  = trans_dy_channel_p2 Bump scale_bump;
////Transfinite Line {offset+213} = trans_dy_channel_p2 Using Progression dy_channel_p2;
//Transfinite Line {offset+213} = trans_dy_channel_p2 Bump scale_bump;
//Transfinite Surface { offset+720025 };



Curve Loop(offset+68001) = {-(offset+82), offset+221, offset+182, -(offset+232)};
Surface (offset+78001) = {offset+68001};
Transfinite Surface {offset+78001};
Surface Loop (offset+88001) = {offset+72000, offset+72004, offset+78001, offset+72003, offset+72002, offset+72001};
Volume(offset+98001) = {offset+88001};
Transfinite Volume {offset+98001};

Curve Loop(offset+68002) = {offset+71, offset+233, -(offset+171), -(offset+201)};
Surface (offset+78002) = {offset+68002};
Transfinite Surface {offset+78002};
Surface Loop (offset+88002) = {offset+78001, offset+72006, offset+78002, offset+72005, offset+72008, offset+72007 };
Volume(offset+98002) = {offset+88002};
Transfinite Volume {offset+98002};

Curve Loop(offset+68003) = {offset+201, -(offset+163), -(offset+204), offset+63};
Surface (offset+78003) = {offset+68003};
Transfinite Surface {offset+78003};
Surface Loop (offset+88003) = {offset+78002, offset+72099, offset+78003, offset+72009, offset+72010};
Volume(offset+98003) = {offset+88003};
Transfinite Volume {offset+98003};

Curve Loop(offset+68004) = {offset+202, offset+151, -(offset+203), -(offset+51)};
Surface (offset+78004) = {offset+68004};
//Transfinite Line {offset+203}  = trans_dy_channel_p1 Using Progression dy_channel_p1;
Transfinite Line {offset+203}  = trans_dy_channel_p1 Using Bump scale_bump;
Transfinite Surface {offset+78004};
Curve Loop(offset+68005) = {offset+62, offset+204, -(offset+162), -(offset+203 )};
Surface (offset+78005) = {offset+68005};
Transfinite Surface {offset+78005};

Surface Loop (offset+88004) = {offset+78003, offset+72011, offset+78004, offset+72012, offset+72015, offset+78005};
Volume(offset+98004) = {offset+88004};
Transfinite Volume {offset+98004};

//Curve Loop(offset+68006) = {offset+203, offset+152, -(offset+212), -(offset+52)};
//Surface (offset+78006) = {offset+68006};
//Transfinite Surface {offset+78006};

//Surface Loop (offset+88005) = {offset+78004, offset+72014, offset+72017,offset+72013, offset+72016, offset+78006};
//Volume(offset+98005) = {offset+88005};
//Transfinite Volume {offset+98005};

Curve Loop(offset+68007) = {offset+203, offset+155, -(offset+206), -(offset+55)};
Surface (offset+78007) = {offset+68007};
Transfinite Surface {offset+78007};

Surface Loop (offset+88006) = {offset+78005, offset+720022, offset+720024, offset+78007, offset+720018, offset+720019};
Volume(offset+98006) = {offset+88006};
Transfinite Volume {offset+98006};

//Surface Loop (offset+88007) = { offset+720023, offset+720021, offset+78007, offset+720020, offset+78006, offset+720025};
//Volume(offset+98007) = {offset+88007};
//Transfinite Volume {offset+98007};

Recombine Surface "*";

Translate {-0.60, -0.0, width} { 
  Surface{:};
}

Rotate {{0, 1, 0}, {0, 0, 0}, Pi/4} {
  Surface{:};
}

Rotate {{1, 0, 0}, {0, 0, 0}, Pi/2} {
  Surface{:};
}

