L = 1.0;

// Points
Point(1) = {0, 0, 0, 1};
Point(2) = {L, 0, 0, 1};
Point(3) = {L, L, 0, 1};
Point(4) = {0, L, 0, 1};

// Lines
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Loop
Line Loop(1) = {1, 2, 3, 4};

// Surface
Plane Surface(1) = {1};