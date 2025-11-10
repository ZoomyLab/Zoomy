scale = 1;
dx = 100/scale;
dy = 10/scale;

x0 = 8;
x1 = 18;
Point(10) = {x0, 0,  0, scale};
Point(11) = {x1, 0,  0, scale};
Point(20) = {x0, 1, 0, scale};
Point(21) = {x1, 1, 0, scale};
Point(30) = {x0, 0,  1, scale};
Point(31) = {x1, 0,  1, scale};
Point(40) = {x0, 1, 1, scale};
Point(41) = {x1, 1, 1, scale};


// bottom
Line(111) = {10, 11};
Line(112) = {20, 21};
Line(113) = {10, 20};
Line(114) = {11, 21};

// top
Line(211) = {30, 31};
Line(212) = {40, 41};
Line(213) = {30, 40};
Line(214) = {31, 41};

// vertical
Line(311) = {10, 30};
Line(312) = {11, 31};
Line(313) = {20, 40};
Line(314) = {21, 41};



Curve Loop(1000) = {111, 114, -112, -113};
Curve Loop(1001) = {211, 214, -212, -213};
Curve Loop(1002) = {111, 312, -211, -311};
Curve Loop(1003) = {112, 314, -212, -313};
Curve Loop(1004) = {113, 313, -213, -311};
Curve Loop(1005) = {114, 314, -214, -312};
Plane Surface(2000) = {1000};
Plane Surface(2001) = {1001};
Plane Surface(2002) = {1002};
Plane Surface(2003) = {1003};
Plane Surface(2004) = {1004};
Plane Surface(2005) = {1005};

Transfinite Line {311, 312, 313, 314 } = 1;
Transfinite Line {111, 112, 211, 212} = dx;
Transfinite Line {114, 214, 113, 213} = dy;

Surface Loop(3000) = {2001, 2002, 2000, 2005, 2003, 2004};
Volume(4000) = {3000};

Transfinite Surface "*";
Transfinite Volume "*";
Recombine Surface "*";
Recombine Volume "*";


Physical Surface("wall", 5002) = {2002, 2003};
Physical Surface("inflow", 5004) = {2004};
Physical Surface("outflow", 5005) = {2005};

Physical Volume("volume", 6000) = {4000};
