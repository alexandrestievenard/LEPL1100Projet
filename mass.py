# =============================================================================
# mass.py
# =============================================================================
# Assemblage de la matrice de masse globale M définie par :
#
#     M_ij = ∫_Ω N_i · N_j dΩ
#
# Elle apparaît dès qu'on discrétise le terme ∂u/∂t. En écrivant
# u_h = Σ_j U_j N_j et en testant avec N_i, on obtient :
#
#     ∫_Ω N_i ∂u_h/∂t dΩ = Σ_j (∫_Ω N_i N_j dΩ) · dU_j/dt = M · dU/dt
#
# M ne dépend ni du temps ni de la solution u → elle est assemblée une
# seule fois avant la boucle temporelle.
#
# Note sur la masse lumpée (utilisée dans imex_solver.py) :
#     M_lump_i = Σ_j M_ij  (somme de la ligne i)
# Cela revient à mettre toute la masse sur la diagonale. L'avantage :
# au lieu de faire M · f (produit matrice-vecteur), on fait M_lump * f
# (multiplication terme à terme), ce qui est bien plus rapide pour
# évaluer la réaction explicite à chaque pas de temps IMEX.
# =============================================================================

import numpy as np
from scipy.sparse import lil_matrix


def assemble_mass(elemTags, conn, det, w, N, tag_to_dof):
    """
    Assemble la matrice de masse globale M.

    L'intégrale M_ij = ∫_Ω N_i N_j dΩ est approchée par quadrature de Gauss,
    élément par élément :

        M_ij ≈ Σ_e Σ_g  w_g · N_i(ξ_g) · N_j(ξ_g) · |det J_e(ξ_g)|

    Paramètres
    ----------
    elemTags  : tags des éléments triangulaires (ne,)
    conn      : connectivité aplatie — tags Gmsh des nœuds (ne*nloc,)
    det       : déterminants des jacobiens aplatis (ne*ngp,)
    w         : poids de quadrature (ngp,)
    N         : valeurs des fonctions de forme aux points de Gauss (ngp*nloc,)
    tag_to_dof: correspondance tag Gmsh → indice DDL compact

    Retour
    ------
    M : lil_matrix (nn * nn) — matrice de masse globale
    """

    # --- Dimensions du problème ---
    ne   = len(elemTags)          # nombre d'éléments
    ngp  = len(w)                 # nombre de points de Gauss par élément
    nloc = int(len(conn) // ne)   # nombre de nœuds locaux par élément (3 pour P1)
    nn   = int(np.max(tag_to_dof) + 1)  # nombre total de DDLs

    # --- Mise en forme des tableaux Gmsh ---
    det  = np.asarray(det,  dtype=np.float64).reshape(ne, ngp)
    conn = np.asarray(conn, dtype=np.int64  ).reshape(ne, nloc)
    N    = np.asarray(N,    dtype=np.float64).reshape(ngp, nloc)

    # Initialisation de la matrice de masse globale (format LIL pour l'assemblage)
    M = lil_matrix((nn, nn), dtype=np.float64)

    # =========================================================================
    # Boucle d'assemblage : élément par élément, point de Gauss par point de Gauss
    # =========================================================================
    for e in range(ne):

        # Indices globaux (compacts) des DDLs de l'élément e
        dof_indices = tag_to_dof[conn[e, :]]

        for g in range(ngp):

            wg   = w[g]        # poids de quadrature
            detg = det[e, g]   # |det J| → change de variable dΩ = det(J) dξ

            # Contribution de ce point de Gauss à toutes les paires (a, b)
            # de fonctions de forme locales :
            # M_ab += w_g · N_a(ξ_g) · N_b(ξ_g) · |det J|
            for a in range(nloc):
                Ia = int(dof_indices[a])
                Na = N[g, a]

                for b in range(nloc):
                    Ib = int(dof_indices[b])
                    M[Ia, Ib] += wg * Na * N[g, b] * detg

    return M