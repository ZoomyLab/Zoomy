scale = 0.125;

x0 = 8;
x1 = 18;
Point(10) = {x0, 0,  0, scale};
Point(11) = {x1, 0,  0, scale};

Point(20) = {x0, 1, 0, scale};
Point(21) = {x1, 1, 0, scale};


Line(111) = {10, 11};
Line(112) = {20, 21};
Line(113) = {10, 20};
Line(114) = {11, 21};


Curve Loop(1000) = {111, 114, -112, -113};
Plane Surface(2000) = {1000};

Transfinite Surface {2000};
Recombine Surface {2000};

Physical Curve("wall", 3000) = {111, 112};
Physical Curve("inflow", 3002) = {113};
Physical Curve("outflow", 3003) = {114};

Physical Surface("surface", 4000) = {2000};
