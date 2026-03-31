LEPL1110 : APE2

Packages you need:
- argparse
- numpy
- matplotlib
- gmsh

Execute in a terminal:
python (or python3) main_poisson_1d.py -order O -cl1 Ha -cl2 Hb -L length

Arguments :
- O : shape functions order (default : 1)
- The mesh follows a linear size field going from Ha to Hb (default : Ha = 0.1 and Hb = 0.1)
- Length : Length of the domain (default : 1.0)

Without modification, the code solves a manufactured problem where the solution is non-polynomial.

For exercice 2, you might need to modify errors.py
For exercice 3, you might need to modify stiffness.py