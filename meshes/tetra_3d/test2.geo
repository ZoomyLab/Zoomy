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
//+
Plane Surface(1) = {202};
//+
Physical Surface("left") = {1};
