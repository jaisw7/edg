lc = 0.5;

p1 = newp; Point(p1) = {0, 0, 0, lc};
p2 = newp; Point(p2) = {1e-3, 0, 0, lc};
p3 = newp; Point(p3) = {1e-3, 1e-3, 0, lc};
p4 = newp; Point(p4) = {0, 1e-3, 0, lc};

l1 = newc; Line(l1) = {p1, p2};
l2 = newc; Line(l2) = {p2, p3};
l3 = newc; Line(l3) = {p3, p4};
l4 = newc; Line(l4) = {p4, p1};

Transfinite Curve{l1} = 3;
Transfinite Curve{l2} = 3;
Transfinite Curve{l3} = 3;
Transfinite Curve{l4} = 3;

Physical Curve("bottom", 5) = {l1};
Physical Curve("right", 6) = {l2};
Physical Curve("top", 7) = {l3};
Physical Curve("left", 8) = {l4};

Curve Loop(1) = {l1, l2, l3, l4};
Plane Surface(1) = {1};
Physical Surface("fluid") = {1};
Transfinite Surface{1} = {l1, l2, l3, l4};

Mesh 2;
Save "cavity_conduction.msh";