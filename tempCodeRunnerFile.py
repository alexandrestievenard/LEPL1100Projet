def build_2d_mesh(L=1.0, lc=1e-2):
    """
    Build and mesh a 2D square [0,L] x [0,L].

    Returns
    -------
    surface : int
        Tag of the plane surface.
    elem_type : int
        Gmsh element type for first-order triangles.
    node_tags : list/array
        Global node tags.
    node_coords : list/array
        Flat array of node coordinates [x1,y1,z1,x2,y2,z2,...].
    elem_tags : list/array
        Element tags.
    elem_node_tags : list/array
        Flat connectivity array.
    """
    gmsh.initialize()
    gmsh.model.add("square_2d")

    # Corner points
    p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc)
    p2 = gmsh.model.geo.addPoint(L,   0.0, 0.0, lc)
    p3 = gmsh.model.geo.addPoint(L,   L,   0.0, lc)
    p4 = gmsh.model.geo.addPoint(0.0, L,   0.0, lc)

    # Boundary lines
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    # Closed boundary of the square
    loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])

    # Surface inside the loop
    surface = gmsh.model.geo.addPlaneSurface([loop])

    gmsh.model.geo.synchronize()

    # Generate 2D mesh
    gmsh.model.mesh.generate(2)

    # Triangle element type, order 1
    elem_type = gmsh.model.mesh.getElementType("triangle", 1)

    # Mesh data
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    elem_tags, elem_node_tags = gmsh.model.mesh.getElementsByType(elem_type)

    return surface, elem_type, node_tags, node_coords, elem_tags, elem_node_tags

def show_mesh(filename="square_2d.msh"):
    """
    Save current mesh and open the Gmsh GUI.
    Assumes a model is already built and meshed.
    """
    gmsh.write(filename)

    gmsh.fltk.run()

    gmsh.finalize()