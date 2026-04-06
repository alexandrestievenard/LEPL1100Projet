# =============================================================================
# plot_utils.py — Utilitaires de visualisation des solutions éléments finis
# =============================================================================
#
# Contient trois fonctions de tracé :
#
#   plot_fe_solution_high_order : tracé 1D haute précision (rééchantillonnage)
#   plot_mesh_2d                : affichage du maillage 2D avec frontières colorées
#   plot_fe_solution_2d         : tracé 2D de la solution par tricontourf
#
# CHOIX DU COLORMAP 'plasma' :
#   Contrairement à 'seismic' (centré sur 0, donne un rose parasite pour u≈0)
#   ou 'YlOrRd' (commence blanc, masque les faibles densités en fond jaune),
#   'plasma' commence dans les tons sombres (noir/violet) pour u=0 et monte
#   vers le jaune vif pour u=vmax. Cela rend le front d'invasion immédiatement
#   visible même pour de très faibles densités.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import gmsh


# -----------------------------------------------------------------------------
# SECTION 1 — Tracé 1D haute précision (solution haute-ordre)
# -----------------------------------------------------------------------------

def plot_fe_solution_high_order(
    elemType, elemNodeTags, nodeCoords, U,
    M=80, show_nodes=False, ax=None, label=None
):
    """
    Trace la solution EF 1D avec rééchantillonnage intra-élément.

    Plutôt que d'afficher simplement les valeurs nodales (ce qui ne montre pas
    la vraie solution polynomiale à l'intérieur des éléments pour les ordres >1),
    on évalue les fonctions de base en M points réguliers sur chaque élément.

    Parameters
    ----------
    elemType    : type d'élément Gmsh (sortie de getElementType)
    elemNodeTags: connectivité aplatie (ne * nloc)
    nodeCoords  : coordonnées nodales aplaties (3 * nn)
    U           : vecteur solution (nn,), aligné sur l'indexation compacte 0..nn-1
    M           : nombre de points d'évaluation par élément (résolution du tracé)
    show_nodes  : si True, affiche aussi les nœuds comme des points ronds
    ax          : axe matplotlib existant (créé si None)
    label       : label pour la légende (affiché sur le premier élément seulement)
    """
    _, _, _, nloc, _, _ = gmsh.model.mesh.getElementProperties(elemType)

    # Points d'évaluation uniformes dans [-1, 1] (coordonnées de référence 1D)
    u_ref = np.linspace(-1.0, 1.0, int(M))
    pts3  = np.zeros((len(u_ref), 3), dtype=float)
    pts3[:, 0] = u_ref
    uvw = pts3.reshape(-1).tolist()

    # Fonctions de base évaluées en ces points
    _, bf, _ = gmsh.model.mesh.getBasisFunctions(elemType, uvw, "Lagrange")
    N = np.asarray(bf, dtype=float).reshape(len(u_ref), nloc)

    if ax is None:
        _, ax = plt.subplots()

    ne = int(len(elemNodeTags) // nloc)
    _, _, coords_flat = gmsh.model.mesh.getJacobians(elemType, uvw)
    coords = np.asarray(coords_flat, dtype=float).reshape(ne, len(u_ref), 3)

    for e in range(ne):
        tags_e = np.asarray(elemNodeTags[e * nloc:(e + 1) * nloc], dtype=int) - 1
        Ue = U[tags_e]
        x  = coords[e, :, 0]
        uh = N @ Ue

        # Trier par x croissant (les éléments ne sont pas forcément ordonnés)
        order = np.argsort(x)
        ax.plot(x[order], uh[order], label=label if (e == 0) else None)

    if show_nodes:
        Xn = np.asarray(nodeCoords, dtype=float).reshape(-1, 3)[:, 0]
        ax.plot(Xn, U, "o", markersize=4)

    ax.set_xlabel("x")
    ax.set_ylabel("u_h")
    ax.grid(True)
    return ax


# -----------------------------------------------------------------------------
# SECTION 2 — Affichage du maillage 2D avec frontières colorées
# -----------------------------------------------------------------------------

def setup_interactive_figure(xlim=None, ylim=None):
    """Crée une figure matplotlib interactive (pour animation en temps réel)."""
    plt.ion()
    fig, ax = plt.subplots()
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    return fig, ax


def plot_mesh_2d(elemType, nodeTags, nodeCoords, elemTags, elemNodeTags,
                 bnds, bnds_tags, tag_to_index=None):
    """
    Affiche le maillage triangulaire 2D et met en valeur les nœuds frontières.

    Pour les éléments d'ordre > 1, seuls les 3 nœuds sommets de chaque triangle
    sont utilisés pour le tracé (Matplotlib ne gère que les triangles P1).

    Parameters
    ----------
    bnds      : liste de (name, dim) — ex: [('OuterBoundary',1), ('Lake',1), ...]
    bnds_tags : liste de tableaux de tags Gmsh, un par frontière
    """
    coords = nodeCoords.reshape(-1, 3)
    x = coords[:, 0]
    y = coords[:, 1]

    # Construction du tableau tag → indice dans nodeCoords
    if tag_to_index is None:
        max_node_tag = int(np.max(nodeTags))
        tag_to_index = np.zeros(max_node_tag + 1, dtype=int)
        for i, tag in enumerate(nodeTags):
            tag_to_index[int(tag)] = i

    # Extraction des 3 nœuds sommets (coins) de chaque triangle
    num_elements  = len(elemTags)
    nodes_per_elem = len(elemNodeTags) // num_elements
    all_nodes     = elemNodeTags.reshape(num_elements, nodes_per_elem)
    corner_nodes  = all_nodes[:, :3]   # les 3 premiers = sommets géométriques
    tri_indices   = tag_to_index[corner_nodes.astype(int)]

    mesh_triang = tri.Triangulation(x, y, tri_indices)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.triplot(mesh_triang, color='black', lw=0.5, alpha=0.4)

    colors = ["red", "darkblue", "orange", "mediumpurple", "pink"]
    for i, (name, dim) in enumerate(bnds):
        tags    = bnds_tags[i]
        indices = tag_to_index[tags.astype(int)]
        ax.scatter(x[indices], y[indices],
                   label=name, s=15, zorder=3,
                   marker="o", facecolor="None",
                   edgecolor=colors[i % len(colors)])

    ax.set_aspect('equal')
    ax.legend(frameon=True, framealpha=1, ncols=2,
              loc="lower center", bbox_to_anchor=(0.5, 1.02))
    plt.axis(False)
    plt.show()


# -----------------------------------------------------------------------------
# SECTION 3 — Tracé 2D de la solution par remplissage de contours
# -----------------------------------------------------------------------------

def plot_fe_solution_2d(elemNodeTags, nodeCoords, nodeTags, U, tag_to_dof,
                        show_mesh=False, ax=None,
                        vmin=0.0, vmax=50.0,
                        cmap='plasma'):
    """
    Trace la solution EF 2D par tricontourf (remplissage de contours triangulés).

    CHOIX TECHNIQUES :
    - vmin=0, vmax=K_RURAL=50 : échelle de couleur fixe et cohérente entre les
      pas de temps, ce qui permet de suivre visuellement la progression du front.
    - extend='max' : les valeurs > vmax (bocage, K=80) gardent la couleur maximale
      sans faire "exploser" l'échelle ni masquer la zone forestière.
    - 256 niveaux de contour : tracé pratiquement continu, sans bandes visibles.

    Parameters
    ----------
    U         : vecteur solution (num_dofs,), indices compacts 0..N-1
    tag_to_dof: tableau de correspondance tag Gmsh → indice DDL
    vmin/vmax : bornes de la colorbar (fixes pour toute la simulation)
    cmap      : colormap Matplotlib (défaut: 'plasma')
    show_mesh : si True, superpose le maillage en blanc fin
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    num_dofs = len(U)

    # Reconstruction des coordonnées dans l'ordre compact (0..num_dofs-1)
    coords_mapped = np.zeros((num_dofs, 2))
    all_coords    = nodeCoords.reshape(-1, 3)
    for i, tag in enumerate(nodeTags):
        dof_idx = tag_to_dof[int(tag)]
        if dof_idx != -1:
            coords_mapped[dof_idx] = all_coords[i, :2]

    x = coords_mapped[:, 0]
    y = coords_mapped[:, 1]

    # Détection automatique du nombre de nœuds par élément
    # (3 pour P1, 6 pour P2, 10 pour P3, 15 pour P4)
    total_nodes = len(elemNodeTags)
    for possible_n in [3, 6, 10, 15, 21]:
        if total_nodes % possible_n == 0:
            nodes_per_elem = possible_n
            break

    # Construction de la triangulation Matplotlib
    # On n'utilise que les 3 nœuds sommets (Matplotlib ne gère que P1)
    conn      = elemNodeTags.reshape(-1, nodes_per_elem)
    triangles = tag_to_dof[conn[:, :3].astype(int)]

    # Remplissage de contours avec 256 niveaux (rendu quasi-continu)
    U_plot  = np.clip(U, vmin, None)   # sécurité : jamais en-dessous de vmin
    contour = ax.tricontourf(
        x, y, triangles, U_plot,
        levels=np.linspace(vmin, vmax, 256),
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        extend='max'   # valeurs > vmax → couleur maximale (pour le bocage K=80)
    )

    if show_mesh:
        ax.triplot(x, y, triangles, color='white', linewidth=0.15, alpha=0.2)

    return contour