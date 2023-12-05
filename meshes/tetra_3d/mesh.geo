Point(1) = {-1, -1, -1, 1.0};
Point(2) = {1, -1, -1, 1.0};
Point(3) = {-1, 1, -1, 1.0};
Point(4) = {1, 1, -1, 1.0};
Point(5) = {-1, -1, 1, 1.0};
Point(6) = {1, -1, 1, 1.0};
Point(7) = {-1, 1, 1, 1.0};
Point(8) = {1, 1, 1, 1.0};

// plane at z = -1
Line(101) = {3, 4};
Line(102) = {4, 2};
Line(103) = {2, 1};
Line(104) = {1, 3};
Line Loop(201) = {-104, -101, -102, -103};

// plane at z = 1
Line(105) = {7, 8};
Line(106) = {8, 6};
Line(107) = {6, 5};
Line(108) = {5, 7};
Line Loop(202) = {108, 105, 106, 107};

// plane at x = -1
Line(109) = {3, 7};
Line(110) = {1, 5};
Line Loop(203) = {104, 109, -108, -110};

// plane at x = 1
Line(111) = {4, 8};
Line(112) = {2, 6};
Line Loop(204) = {102, -111, -106, 112};

// plane at y = -1
Line Loop(205) = {110, -107, -112, 103};

// plane at y = 1
Line Loop(206) = {-109, -105, 111, 101};


Plane Surface(301) = {201};
Plane Surface(302) = {202};
Plane Surface(303) = {203};
Plane Surface(304) = {204};
Plane Surface(305) = {205};
Plane Surface(306) = {206};

Surface Loop(401) = {301, 302, 303, 304, 305, 306};
Volume(501) = {401};

// Transfinite Surface {301};
// Transfinite Surface {302};
// Transfinite Surface {303};
// Transfinite Surface {304};
// Transfinite Surface {305};
// Transfinite Surface {306};
// Recombine Surface {301};
// Recombine Surface {302};
// Recombine Surface {303};
// Recombine Surface {304};
// Recombine Surface {305};
// Recombine Surface {306};

// Recombine Surface {301, 302, 303, 304, 305, 306};
// Transfinite Volume {501};

Physical Surface("bottom", 1001) = {301};
Physical Surface("top", 1002) = {302};
Physical Surface("left", 1003) = {303};
Physical Surface("right", 1004) = {304};
Physical Surface("front", 1005) = {305};
Physical Surface("back", 1006) = {306};
Physical Volume("volume", 2001) = {501};
