import gmsh
import math

gmsh.initialize()
gmsh.model.add("rect_with_hole")

Lx, Ly = 4.0, 2.0
cx, cy = 2.0, 1.0
r = 0.3

rect = gmsh.model.occ.addRectangle(0, 0, 0, Lx, Ly)
disk = gmsh.model.occ.addDisk(cx, cy, 0, 2*r, r)

# Remove disk from rect
out, _ = gmsh.model.occ.cut([(2, rect)], [(2, disk)], removeObject=True, removeTool=True) 
gmsh.model.occ.synchronize()

# Resulting domain surface
domain = gmsh.model.getEntities(2)
domain_tag = domain[0][1]


gmsh.model.addPhysicalGroup(2, [domain_tag], tag=1)
gmsh.model.setPhysicalName(2, 1, "Domain")

boundary = gmsh.model.getBoundary([(2, domain_tag)], oriented=False)

outer_curves = []
inner_curves = []

# Determine outer and inner boundaries
for dim, tag in boundary:
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
    mx = 0.5 * (xmin + xmax)
    my = 0.5 * (ymin + ymax)

    # If midpoint is near circle center, treat as inner boundary
    dist = math.sqrt((mx - cx)**2 + (my - cy)**2)

    if dist < 1.5 * r:
        inner_curves.append(tag)
    else:
        outer_curves.append(tag)

print("Outer curves:", outer_curves)
print("Inner curves:", inner_curves)

gmsh.model.addPhysicalGroup(1, outer_curves, tag=2)
gmsh.model.setPhysicalName(1, 2, "OuterBoundary")

gmsh.model.addPhysicalGroup(1, inner_curves, tag=3)
gmsh.model.setPhysicalName(1, 3, "InnerBoundary")

gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.05)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.15)

gmsh.model.mesh.generate(2)
gmsh.write("rect_hole.msh")
gmsh.finalize()