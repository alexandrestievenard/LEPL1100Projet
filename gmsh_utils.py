# =============================================================================
# gmsh_utils.py
# =============================================================================
# Couche d'interface entre Gmsh et notre solveur éléments finis.
#
# Gmsh génère le maillage et fournit toutes les données géométriques, mais
# dans ses propres formats (tableaux aplatis, tags non contigus, etc.).
# Ce fichier traduit ces données en structures directement utilisables par
# les routines d'assemblage (mass.py, stiffness_non_linear.py, ...).
#
# VOCABULAIRE GMSH :
#   tag       : identifiant interne Gmsh d'un nœud (commence à 1, peut avoir
#               des trous : 1, 2, 5, 7, ...)
#   dof index : indice compact 0..N-1 dans nos matrices (toujours contigu)
#   tag_to_dof: tableau de correspondance tag → indice DDL
#               (construit dans build_problem() de runsimulation.py)
#   elemType  : entier Gmsh identifiant le type d'élément (ligne, triangle...)
#
# FONCTIONS UTILISÉES DANS LE PROJET ACTUEL :
#   gmsh_init, gmsh_finalize     → cycle de vie de Gmsh
#   open_2d_mesh                 → chargement du maillage Corse
#   prepare_quadrature_and_basis → points et poids de Gauss + fonctions de base
#   get_jacobians                → jacobiens de la transformation référence→physique
#   border_dofs_from_tags        → conversion tags frontière → indices DDL
#
# FONCTIONS HÉRITÉES (non utilisées dans le projet Corse) :
#   build_1d_mesh, end_dofs_from_nodes, getPhysical
#   (conservées pour compatibilité avec d'autres codes du cours)
# =============================================================================

import numpy as np
import gmsh


# =============================================================================
# SECTION 1 — Cycle de vie de Gmsh
# =============================================================================

def gmsh_init(model_name="fem"):
    """
    Initialise la bibliothèque Gmsh et crée un modèle vide.
    Doit être appelée une seule fois avant toute opération Gmsh.
    """
    gmsh.initialize()
    gmsh.model.add(model_name)


def gmsh_finalize():
    """
    Libère toutes les ressources Gmsh.
    Doit être appelée en fin de programme pour éviter les fuites mémoire.
    """
    gmsh.finalize()


# =============================================================================
# SECTION 2 — Construction d'un maillage 1D
# =============================================================================

def build_1d_mesh(L=1.0, cl1=0.02, cl2=0.10, order=1):
    """
    Construit et maille le segment [0, L] avec tailles de maille variables.

    cl1 et cl2 contrôlent la finesse du maillage aux deux extrémités.
    Gmsh interpole automatiquement la taille entre les deux points.

    Paramètres
    ----------
    L    : longueur du segment
    cl1  : taille de maille caractéristique en x=0
    cl2  : taille de maille caractéristique en x=L
    order: ordre polynomial des éléments (1=linéaire, 2=quadratique)

    Retour
    ------
    line, elemType, nodeTags, nodeCoords, elemTags, elemNodeTags
    """
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


# =============================================================================
# SECTION 3 — Chargement d'un maillage 2D depuis un fichier .msh
# =============================================================================

def open_2d_mesh(msh_filename, order=1):
    """
    Charge un fichier .msh et retourne toutes les données nécessaires au solveur.

    Les frontières sont lues via les groupes physiques 1D définis dans le
    fichier (ex: OuterBoundary, Mountains). Cela évite de devoir "deviner"
    les frontières par topologie, ce qui était fragile dans les versions
    précédentes du code.

    Paramètres
    ----------
    msh_filename : chemin vers le fichier .msh (ex: "invasion_map.msh")
    order        : ordre polynomial des éléments (1 ou 2)

    Retour
    ------
    elemType     : type d'élément triangle Gmsh (entier)
    nodeTags     : tags Gmsh de tous les nœuds (nn,)
    nodeCoords   : coordonnées [x,y,z] aplaties (3*nn,)
    elemTags     : tags des éléments triangulaires (ne,)
    elemNodeTags : connectivité aplatie — tags Gmsh des nœuds (ne*nloc,)
    bnds         : liste de (nom, dim) pour chaque frontière physique
                   ex: [('OuterBoundary', 1), ('Mountains', 1)]
    bnds_tags    : liste de tableaux de tags Gmsh, un par frontière
                   (même ordre que bnds)
    """
    gmsh.open(msh_filename)
    gmsh.model.mesh.setOrder(order)

    # Éléments triangulaires du domaine de calcul (dim=2)
    elemType                = gmsh.model.mesh.getElementType("triangle", order)
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    elemTags, elemNodeTags  = gmsh.model.mesh.getElementsByType(elemType)

    # Lecture des groupes physiques 1D = les frontières nommées du domaine.
    # Dans msh.py, on a défini OuterBoundary (côte) et Mountains (massifs).
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
        # getNodesForPhysicalGroup retourne les tags Gmsh de tous les nœuds
        # appartenant à cette frontière physique
        node_tags_bnd, _ = gmsh.model.mesh.getNodesForPhysicalGroup(dim, tag)
        bnds.append((name, dim))
        bnds_tags.append(node_tags_bnd)

    print(f"Frontières chargées depuis '{msh_filename}':")
    for (name, _), tags in zip(bnds, bnds_tags):
        print(f"  [{name}] — {len(tags)} nœuds")

    return elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags


# =============================================================================
# SECTION 4 — Quadrature de Gauss et fonctions de base
# =============================================================================

def prepare_quadrature_and_basis(elemType, order):
    """
    Prépare les points et poids de quadrature de Gauss, ainsi que les
    valeurs des fonctions de forme et de leurs gradients en ces points.

    Pourquoi la quadrature de Gauss ?
    ----------------------------------
    Plutôt que d'intégrer analytiquement sur chaque triangle physique
    (fastidieux et spécifique à chaque géométrie), on ramène tout à un
    triangle de référence unique. La quadrature de Gauss donne des
    points ξ_g et des poids w_g tels que :

        ∫_ref f(ξ) dξ ≈ Σ_g w_g · f(ξ_g)

    avec exactitude pour les polynômes jusqu'au degré voulu.
    La règle "Gauss{2*order}" intègre exactement les polynômes de degré
    2*order, ce qui est suffisant pour les termes bilinéaires de K et M
    (produits de deux polynômes de degré `order`).

    Paramètres
    ----------
    elemType : type d'élément Gmsh
    order    : ordre polynomial des éléments

    Retour
    ------
    xi : coordonnées de référence des points de Gauss (aplaties)
    w  : poids de quadrature (ngp,)
    N  : valeurs des fonctions de forme aux points de Gauss (aplaties, ngp*nloc)
    gN : gradients des fonctions de forme en coordonnées de référence (aplatis, ngp*nloc*3)
    """
    rule = f"Gauss{2 * order}"

    # Points et poids de Gauss dans l'élément de référence
    xi, w = gmsh.model.mesh.getIntegrationPoints(elemType, rule)

    # Valeurs des fonctions de forme N_a(ξ_g) aux points de Gauss
    _, N,  _ = gmsh.model.mesh.getBasisFunctions(elemType, xi, "Lagrange")

    # Gradients ∇_ξ N_a(ξ_g) en coordonnées de référence.
    # Pour obtenir les gradients physiques ∇_x N_a, il faudra appliquer
    # J^{-T} dans les routines d'assemblage (stiffness_non_linear.py).
    _, gN, _ = gmsh.model.mesh.getBasisFunctions(elemType, xi, "GradLagrange")

    return xi, np.asarray(w, dtype=float), N, gN


# =============================================================================
# SECTION 5 — Jacobiens de la transformation référence → physique
# =============================================================================

def get_jacobians(elemType, xi, tag=-1):
    """
    Calcule les jacobiens de la transformation référence → physique pour
    tous les éléments du type donné.

    Pourquoi le jacobien ?
    ----------------------
    Chaque élément triangulaire physique est l'image du triangle de référence
    par une transformation affine x = x(ξ). Le jacobien J de cette
    transformation sert à deux choses :

      1. Changement de variable dans les intégrales :
             ∫_e f(x) dΩ = ∫_ref f(x(ξ)) |det J(ξ)| dξ

      2. Conversion des gradients de référence en gradients physiques :
             ∇_x N_a = J^{-T} ∇_ξ N_a

    Gmsh calcule J et det(J) pour nous à chaque point de Gauss.

    Paramètres
    ----------
    elemType : type d'élément Gmsh
    xi       : coordonnées de référence des points de Gauss
               (sortie de prepare_quadrature_and_basis)
    tag      : tag de l'entité géométrique (-1 = tous les éléments du type)

    Retour
    ------
    jac    : jacobiens aplatis (ne * ngp * 9)  — matrice 3×3 par pt de Gauss
    dets   : déterminants aplatis (ne * ngp)   — |det J| par pt de Gauss
    coords : coordonnées physiques aplaties (ne * ngp * 3) — x_g dans l'espace réel
    """
    jac, dets, coords = gmsh.model.mesh.getJacobians(elemType, xi, tag=tag)
    return jac, dets, coords


# =============================================================================
# SECTION 6 — Conversion tags Gmsh ↔ indices DDL
# =============================================================================

def end_dofs_from_nodes(nodeCoords):
    """
    Identifie les DDLs aux extrémités d'un segment 1D (x_min et x_max).
    Utilisée pour imposer les conditions de Dirichlet en 1D.

    Retour
    ------
    left_dof, right_dof : indices 0-based dans nos matrices
    """
    X     = np.asarray(nodeCoords, dtype=float).reshape(-1, 3)[:, 0]
    left  = int(np.argmin(X))
    right = int(np.argmax(X))
    return left, right


def border_dofs_from_tags(l_tags, tag_to_dof):
    """
    Convertit des tags Gmsh d'une frontière physique en indices compacts
    de DDLs utilisables dans nos matrices globales.

    Gmsh numérote ses nœuds avec des tags qui peuvent commencer à 1 et
    avoir des trous. Nos matrices utilisent des indices contigus 0..N-1.
    Ce tableau de correspondance tag_to_dof fait le lien entre les deux.

    Les tags dont tag_to_dof[tag] == -1 sont filtrés : ce sont des nœuds
    géométriques purs (coins de la géométrie Gmsh) qui n'ont pas de DDL
    associé dans le problème EF.

    Paramètres
    ----------
    l_tags     : tags Gmsh des nœuds d'une frontière (sortie de getNodesForPhysicalGroup)
    tag_to_dof : tableau de correspondance tag → indice DDL

    Retour
    ------
    l_dofs : tableau d'indices DDL compacts correspondants
    """
    l_tags     = np.asarray(l_tags, dtype=int)
    valid_mask = (tag_to_dof[l_tags] != -1)   # filtre les tags sans DDL associé
    l_dofs     = tag_to_dof[l_tags[valid_mask]]
    return l_dofs


def getPhysical(name):
    """
    Récupère les éléments et nœuds d'un groupe physique par son nom.
    Utile pour les intégrales de bord (termes de Neumann non homogènes).

    Retour
    ------
    elemType, elemTags, elemNodeTags, entityTag
    """
    dimTags = gmsh.model.getEntitiesForPhysicalName(name)
    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(
        dim=dimTags[0][0], tag=dimTags[0][1]
    )
    return elemTypes[0], elemTags[0], elemNodeTags[0], dimTags[0][1]