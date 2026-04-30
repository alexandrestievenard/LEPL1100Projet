import numpy as np
import gmsh

def gmsh_init(model_name="fem"):
    gmsh.initialize()
    gmsh.model.add(model_name)
def gmsh_finalize():
    gmsh.finalize()
    

def build_1d_mesh(L=1.0, cl1=0.02, cl2=0.10, order=1):

    p0   = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, cl1)
    p1   = gmsh.model.geo.addPoint(L,   0.0, 0.0, cl2)
    line = gmsh.model.geo.addLine(p0, p1)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(1)
    gmsh.model.mesh.setOrder(order)

    elemType               = gmsh.model.mesh.getElementType("line", order)
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    elemTags, elemNodeTags  = gmsh.model.mesh.getElementsByType(elemType)

    return line, elemType, nodeTags, nodeCoords, elemTags, elemNodeTags

def open_2d_mesh(msh_filename, order=1):
    gmsh.open(msh_filename)
    gmsh.model.mesh.setOrder(order)

    elemType                = gmsh.model.mesh.getElementType("triangle", order)
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    elemTags, elemNodeTags  = gmsh.model.mesh.getElementsByType(elemType)
    phys_groups_1d = gmsh.model.getPhysicalGroups(dim=1)

    if len(phys_groups_1d) == 0:
        raise RuntimeError(
            f"Aucun groupe physique 1D trouvé dans '{msh_filename}'.\n"
            "Le fichier .msh doit être généré avec msh.py, qui définit\n"
            "les groupes physiques nommés (OuterBoundary, Mountains)."
        )

    bnds      = []
    bnds_tags = []

    for dim, tag in phys_groups_1d:
        name = gmsh.model.getPhysicalName(dim, tag)
        node_tags_bnd, _ = gmsh.model.mesh.getNodesForPhysicalGroup(dim, tag)
        bnds.append((name, dim))
        bnds_tags.append(node_tags_bnd)

    print(f"Frontières chargées depuis '{msh_filename}':")
    for (name, _), tags in zip(bnds, bnds_tags):
        print(f"  [{name}] — {len(tags)} nœuds")

    return elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags

def prepare_quadrature_and_basis(elemType, order):
    rule = f"Gauss{2 * order}"

    xi, w = gmsh.model.mesh.getIntegrationPoints(elemType, rule)

    _, N,  _ = gmsh.model.mesh.getBasisFunctions(elemType, xi, "Lagrange")
    _, gN, _ = gmsh.model.mesh.getBasisFunctions(elemType, xi, "GradLagrange")
    return xi, np.asarray(w, dtype=float), N, gN

def get_jacobians(elemType, xi, tag=-1):
    jac, dets, coords = gmsh.model.mesh.getJacobians(elemType, xi, tag=tag)
    return jac, dets, coords

def end_dofs_from_nodes(nodeCoords):
    X     = np.asarray(nodeCoords, dtype=float).reshape(-1, 3)[:, 0]
    left  = int(np.argmin(X))
    right = int(np.argmax(X))
    return left, right


def border_dofs_from_tags(l_tags, tag_to_dof):
    l_tags     = np.asarray(l_tags, dtype=int)
    valid_mask = (tag_to_dof[l_tags] != -1)  
    l_dofs     = tag_to_dof[l_tags[valid_mask]]
    return l_dofs


def getPhysical(name):
    dimTags = gmsh.model.getEntitiesForPhysicalName(name)
    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim=dimTags[0][0], tag=dimTags[0][1])
    return elemTypes[0], elemTags[0], elemNodeTags[0], dimTags[0][1]