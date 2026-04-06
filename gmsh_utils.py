# =============================================================================
# gmsh_utils.py — Utilitaires d'interface avec la bibliothèque Gmsh
# =============================================================================
#
# Ce fichier contient toutes les fonctions d'interaction bas-niveau avec Gmsh :
#   - Initialisation / finalisation
#   - Construction et chargement de maillages (1D et 2D)
#   - Préparation des points de quadrature et des fonctions de base
#   - Calcul des jacobiens
#   - Conversion des tags Gmsh en indices de degrés de liberté (DDLs)
#
# VOCABULAIRE GMSH IMPORTANT :
#   - "node tag" : identifiant interne Gmsh d'un nœud (commence à 1, peut avoir des trous)
#   - "dof index" : indice compact 0..N-1 dans nos matrices globales (contigu)
#   - "tag_to_dof" : tableau de correspondance tag → indice DDL
#   - "elemType" : entier Gmsh identifiant le type d'élément (ligne, triangle, etc.)
# =============================================================================

import numpy as np
import gmsh


# -----------------------------------------------------------------------------
# SECTION 1 — Gestion du cycle de vie de Gmsh
# -----------------------------------------------------------------------------

def gmsh_init(model_name="fem"):
    """Initialise Gmsh et crée un nouveau modèle vide."""
    gmsh.initialize()
    gmsh.model.add(model_name)


def gmsh_finalize():
    """Libère toutes les ressources Gmsh. À appeler en fin de programme."""
    gmsh.finalize()


# -----------------------------------------------------------------------------
# SECTION 2 — Construction d'un maillage 1D (pour main_diffusion_1d.py)
# -----------------------------------------------------------------------------

def build_1d_mesh(L=1.0, cl1=0.02, cl2=0.10, order=1):
    """
    Construit et maille le segment [0, L] avec des tailles de maille variables.

    Parameters
    ----------
    L    : longueur du segment
    cl1  : taille de maille au point x=0
    cl2  : taille de maille au point x=L
    order: ordre polynomial des éléments (1=linéaire, 2=quadratique, ...)

    Returns
    -------
    line, elemType, nodeTags, nodeCoords, elemTags, elemNodeTags
    """
    p0 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, cl1)
    p1 = gmsh.model.geo.addPoint(L,   0.0, 0.0, cl2)
    line = gmsh.model.geo.addLine(p0, p1)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(1)
    gmsh.model.mesh.setOrder(order)

    elemType = gmsh.model.mesh.getElementType("line", order)
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    elemTags, elemNodeTags  = gmsh.model.mesh.getElementsByType(elemType)

    return line, elemType, nodeTags, nodeCoords, elemTags, elemNodeTags


# -----------------------------------------------------------------------------
# SECTION 3 — Chargement d'un maillage 2D depuis un fichier .msh
# -----------------------------------------------------------------------------

def open_2d_mesh(msh_filename, order=1):
    """
    Charge un fichier .msh et retourne toutes les données nécessaires au solveur.

    Contrairement à l'ancienne version qui "devinait" les frontières par topologie,
    cette version lit directement les groupes physiques 1D définis dans le fichier
    (OuterBoundary, Lake, Mountains, ...). Cela la rend robuste et générique :
    elle fonctionne avec n'importe quel nombre de frontières nommées.

    Parameters
    ----------
    msh_filename : str — chemin vers le fichier .msh (ex: "invasion_map.msh")
    order        : int — ordre polynomial des éléments (1 ou 2)

    Returns
    -------
    elemType     : int           — type d'élément triangle Gmsh
    nodeTags     : array (nn,)   — tags Gmsh des nœuds
    nodeCoords   : array (3*nn,) — coordonnées [x,y,z] aplaties
    elemTags     : array (ne,)   — tags des éléments triangulaires
    elemNodeTags : array (ne*nloc,) — connectivité (tags Gmsh)
    bnds         : list of (name: str, dim: int)
                   ex: [('OuterBoundary',1), ('Lake',1), ('Mountains',1)]
    bnds_tags    : list of array — tags des nœuds de chaque frontière
                   (même ordre que bnds)
    """
    gmsh.open(msh_filename)
    gmsh.model.mesh.setOrder(order)

    # ── Éléments triangulaires du domaine 2D ──────────────────────────────
    elemType     = gmsh.model.mesh.getElementType("triangle", order)
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    elemTags, elemNodeTags  = gmsh.model.mesh.getElementsByType(elemType)

    # ── Lecture des groupes physiques 1D (= frontières nommées) ──────────
    phys_groups_1d = gmsh.model.getPhysicalGroups(dim=1)

    if len(phys_groups_1d) == 0:
        raise RuntimeError(
            f"Aucun groupe physique 1D trouvé dans '{msh_filename}'.\n"
            "Le fichier .msh doit être généré avec msh.py qui définit les groupes\n"
            "physiques nommés (OuterBoundary, Lake, Mountains)."
        )

    bnds      = []
    bnds_tags = []

    for dim, tag in phys_groups_1d:
        name = gmsh.model.getPhysicalName(dim, tag)
        node_tags_bnd, _ = gmsh.model.mesh.getNodesForPhysicalGroup(dim, tag)
        bnds.append((name, dim))
        bnds_tags.append(node_tags_bnd)

    # Affichage de contrôle
    print(f"Frontières chargées depuis '{msh_filename}':")
    for (name, _), tags in zip(bnds, bnds_tags):
        print(f"  [{name}] — {len(tags)} nœuds")

    return elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags


# -----------------------------------------------------------------------------
# SECTION 4 — Quadrature et fonctions de base
# -----------------------------------------------------------------------------

def prepare_quadrature_and_basis(elemType, order):
    """
    Prépare les points de quadrature de Gauss et les valeurs des fonctions de
    base (et de leurs gradients en coordonnées de référence) en ces points.

    La règle "Gauss{2*order}" est exacte pour les polynômes de degré 2*order,
    ce qui garantit l'intégration exacte des termes bilinéaires (K et M).

    Returns
    -------
    xi  : coordonnées de référence des points de Gauss (aplaties)
    w   : poids de quadrature (ngp,)
    N   : valeurs des fonctions de base aux pts de Gauss (ngp*nloc, aplaties)
    gN  : gradients des fonctions de base (référence) (ngp*nloc*3, aplaties)
    """
    rule = f"Gauss{2 * order}"
    xi, w = gmsh.model.mesh.getIntegrationPoints(elemType, rule)
    _, N,  _ = gmsh.model.mesh.getBasisFunctions(elemType, xi, "Lagrange")
    _, gN, _ = gmsh.model.mesh.getBasisFunctions(elemType, xi, "GradLagrange")
    return xi, np.asarray(w, dtype=float), N, gN


# -----------------------------------------------------------------------------
# SECTION 5 — Jacobiens
# -----------------------------------------------------------------------------

def get_jacobians(elemType, xi, tag=-1):
    """
    Calcule les jacobiens de la transformation de référence → physique.

    Le jacobien J(x) permet de :
      1. Calculer det(J) pour les intégrales : ∫_e f dx = ∫_ref f det(J) dξ
      2. Calculer inv(J) pour les gradients physiques : ∇N = inv(J)^T ∇_ref N

    Parameters
    ----------
    elemType : type d'élément Gmsh
    xi       : coordonnées de référence des points de Gauss (sortie de prepare_quadrature_and_basis)
    tag      : tag de l'entité (-1 = tous les éléments du type)

    Returns
    -------
    jac    : jacobiens aplatis (ne * ngp * 9)
    dets   : déterminants aplatis (ne * ngp)
    coords : coordonnées physiques des pts de Gauss aplaties (ne * ngp * 3)
    """
    jac, dets, coords = gmsh.model.mesh.getJacobians(elemType, xi, tag=tag)
    return jac, dets, coords


# -----------------------------------------------------------------------------
# SECTION 6 — Conversion tags Gmsh ↔ indices DDL
# -----------------------------------------------------------------------------

def end_dofs_from_nodes(nodeCoords):
    """
    Identifie les DDLs aux extrémités d'un segment 1D (x_min et x_max).
    Utilisé par main_diffusion_1d.py pour imposer les conditions Dirichlet.

    Returns
    -------
    left_dof, right_dof : indices 0-based dans nos matrices
    """
    X     = np.asarray(nodeCoords, dtype=float).reshape(-1, 3)[:, 0]
    left  = int(np.argmin(X))
    right = int(np.argmax(X))
    return left, right


def border_dofs_from_tags(l_tags, tag_to_dof):
    """
    Convertit une liste de tags Gmsh (d'une frontière physique) en indices
    compacts de DDLs (0..N-1) utilisables dans nos matrices.

    Les tags invalides (tag_to_dof[tag] == -1) sont filtrés automatiquement.
    Cela peut arriver pour des nœuds géométriques purs qui ne correspondent
    pas à des DDLs du problème.

    Parameters
    ----------
    l_tags     : array de tags Gmsh (sortie de getNodesForPhysicalGroup)
    tag_to_dof : array de correspondance tag → indice DDL

    Returns
    -------
    l_dofs : array d'indices DDL (dtype int)
    """
    l_tags     = np.asarray(l_tags, dtype=int)
    valid_mask = (tag_to_dof[l_tags] != -1)
    l_dofs     = tag_to_dof[l_tags[valid_mask]]
    return l_dofs


def getPhysical(name):
    """
    Récupère les éléments et nœuds d'un groupe physique par son nom.
    Utilisé pour les intégrales de bord (termes Neumann).

    Returns
    -------
    elemType, elemTags, elemNodeTags, entityTag
    """
    dimTags    = gmsh.model.getEntitiesForPhysicalName(name)
    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(
        dim=dimTags[0][0], tag=dimTags[0][1]
    )
    return elemTypes[0], elemTags[0], elemNodeTags[0], dimTags[0][1]