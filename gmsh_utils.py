# gmsh_utils.py
import numpy as np
import gmsh


def gmsh_init(model_name="fem1d"):
    gmsh.initialize()
    gmsh.model.add(model_name)


def gmsh_finalize():
    gmsh.finalize()


def build_1d_mesh(L=1.0, cl1=0.02, cl2=0.10, order=1):
    """
    Build and mesh a 1D segment [0,L] with different characteristic lengths.
    Returns (line_tag, elemType, nodeTags, nodeCoords, elemTags, elemNodeTags).
    """
    p0 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, cl1)
    p1 = gmsh.model.geo.addPoint(L, 0.0, 0.0, cl2)
    line = gmsh.model.geo.addLine(p0, p1)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(1)
    gmsh.model.mesh.setOrder(order)

    elemType = gmsh.model.mesh.getElementType("line", order)

    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    elemTags, elemNodeTags = gmsh.model.mesh.getElementsByType(elemType)

    return line, elemType, nodeTags, nodeCoords, elemTags, elemNodeTags


def prepare_quadrature_and_basis(elemType, order):
    """
    Returns:
      xi (flattened uvw), w (ngp), N (flattened bf), gN (flattened gbf)
    """
    rule = f"Gauss{2 * order}"
    xi, w = gmsh.model.mesh.getIntegrationPoints(elemType, rule)
    _, N, _ = gmsh.model.mesh.getBasisFunctions(elemType, xi, "Lagrange")
    _, gN, _ = gmsh.model.mesh.getBasisFunctions(elemType, xi, "GradLagrange")
    return xi, np.asarray(w, dtype=float), N, gN


def get_jacobians(elemType, xi):
    """
    Wrapper around gmsh.getJacobians.
    Returns (jacobians, dets, coords)
    """
    jacobians, dets, coords = gmsh.model.mesh.getJacobians(elemType, xi)
    return jacobians, dets, coords


def end_dofs_from_nodes(nodeCoords):
    """
    Robustly identify first/last node dofs from coordinates (x-min, x-max).
    nodeCoords is flattened [x0,y0,z0, x1,y1,z1, ...]
    Returns (left_dof, right_dof) as 0-based indices.
    """
    X = np.asarray(nodeCoords, dtype=float).reshape(-1, 3)[:, 0]
    left = int(np.argmin(X))
    right = int(np.argmax(X))
    return left, right


def build_2d_mesh(geo_filename, mesh_size, order=1):
    """
    Load a .geo file and generate a 2D mesh with uniform element size.

    Parameters
    ----------
    geo_filename : str
        Path to the .geo file
    mesh_size : float
        Target mesh size (uniform)
    order : int
        Polynomial order of elements

    Returns
    -------
    elemType, nodeTags, nodeCoords, elemTags, elemNodeTags
    """

    import gmsh

    # --- load geometry
    gmsh.open(geo_filename)

    # --- FORCE uniform mesh size everywhere
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)

    # prevent boundary propagation (VERY important)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    # disable curvature & point based sizing
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    # --- generate 2D mesh
    gmsh.model.mesh.generate(2)

    # --- high order
    gmsh.model.mesh.setOrder(order)

    # --- element type (triangles)
    elemType = gmsh.model.mesh.getElementType("triangle", order)

    # --- nodes
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()

    # --- elements
    elemTags, elemNodeTags = gmsh.model.mesh.getElementsByType(elemType)

    return elemType, nodeTags, nodeCoords, elemTags, elemNodeTags

def format_2d_mesh(elemType, nodeTags, nodeCoords, elemTags, elemNodeTags):
    """
    Helper function to convert raw arrays from build_2d_mesh in easier format:
    - coords: [(x1,y1,z1), (x2,y2,z3), ...]
    - elements: [(e1tag1,e1tag2,e1tag3), (e2tag1,e2tag2,e2tag3), ...]
    - elements_idx: [(e1idx1,e1idx2,e1idx3), ...]

    Example : to know coordinates of nodes of first triangle we can simply do:
    coords[elements_idx[0]]

    with x1 = coords[elements_idx[0][0]]
    with x2 = coords[elements_idx[0][1]]
    with x3 = coords[elements_idx[0][2]]
    """
    # Reshape arrays to convenient format
    coords = nodeCoords.reshape(-1,3)
    elements = elemNodeTags.reshape(-1,3)

    # Convert tags to indices
    tag_to_index = {tag: i for i, tag in enumerate(nodeTags)}
    
    # Convert elements[tag] to elements[idx] where idx is index in coords
    elements_idx = [
    [tag_to_index[tag] for tag in triangle]
    for triangle in elements
    ]
    return coords, elements, elements_idx
