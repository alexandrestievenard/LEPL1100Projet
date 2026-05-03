# Assemblage de la matrice de rigidité K et du vecteur de charge F pour
# l'équation de diffusion avec diffusivité non linéaire :
#
#     -div( κ(u,x) ∇u ) = f(x)
#
# La formulation faible discrétisée donne le système K·U = F avec :
#
#     K_ab = ∫_Ω κ(u_h, x) ∇N_a · ∇N_b dΩ
#     F_a  = ∫_Ω f(x) N_a dΩ
#
# La particularité par rapport au cas linéaire (stiffness.py) est que κ
# dépend de u_h, la solution EF courante. On doit donc reconstruire u_h
# aux points de Gauss avant d'évaluer κ.
#
# Dans le schéma IMEX, cette fonction est appelée avec U = u^n (connu),
# ce qui fait de κ(u^n, x) une simple fonction d'espace : le système
# résultant reste linéaire en u^{n+1}.

import numpy as np
from scipy.sparse import lil_matrix


def assemble_stiffness_and_rhs(elemTags, conn, jac, det, xphys, w, N, gN, U, kappa_fun, rhs_fun, tag_to_dof):
    """
    Assemble la matrice de rigidité K et le vecteur de charge F.

    Paramètres
    ----------
    elemTags   : tags des éléments triangulaires (ne,)
    conn       : connectivité aplatie — tags Gmsh des nœuds de chaque élément (ne*nloc,)
    jac        : jacobiens de la transformation référence→physique (ne*ngp*9,)
    det        : déterminants des jacobiens (ne*ngp,)
    xphys      : coordonnées physiques des points de Gauss (ne*ngp*3,)
    w          : poids de quadrature (ngp,)
    N          : valeurs des fonctions de forme aux points de Gauss (ngp*nloc,)
    gN         : gradients des fonctions de forme en coordonnées de référence (ngp*nloc*3,)
    U          : solution nodale courante u^n (nn,)
    kappa_fun  : diffusivité κ(u, x) — fonction à 2 arguments
    rhs_fun    : terme source f(x) — fonction à 1 argument
    tag_to_dof : correspondance tag Gmsh → indice DDL compact

    Retour
    ------
    K : lil_matrix (nn × nn) — matrice de rigidité
    F : ndarray   (nn,)      — vecteur de charge
    """

    ne   = len(elemTags)
    ngp  = len(w)
    nloc = int(len(conn) // ne)
    nn   = int(np.max(tag_to_dof) + 1)

    # Gmsh renvoie tout aplati ; on remet en forme (ne, ngp, ...) pour indexer proprement
    det   = np.asarray(det,   dtype=np.float64).reshape(ne, ngp)
    xphys = np.asarray(xphys, dtype=np.float64).reshape(ne, ngp, 3)
    jac   = np.asarray(jac,   dtype=np.float64).reshape(ne, ngp, 3, 3)
    conn  = np.asarray(conn,  dtype=np.int64  ).reshape(ne, nloc)
    N     = np.asarray(N,     dtype=np.float64).reshape(ngp, nloc)
    gN    = np.asarray(gN,    dtype=np.float64).reshape(ngp, nloc, 3)

    K = lil_matrix((nn, nn), dtype=np.float64)
    F = np.zeros(nn, dtype=np.float64)

    for e in range(ne):

        dof_indices = tag_to_dof[conn[e, :]]
        Ue = U[dof_indices]

        for g in range(ngp):

            xg   = xphys[e, g]
            wg   = w[g]
            detg = det[e, g]

            # Gmsh donne J tel que dx = J dξ, donc ∇_x N = J^{-T} ∇_ξ N
            invjacg = np.linalg.inv(jac[e, g])

            # u_h(ξ_g) = Σ_a U_a^e · N_a(ξ_g)
            u_g     = float(np.dot(Ue, N[g, :]))
            kappa_g = float(kappa_fun(u_g, xg))
            f_g     = float(rhs_fun(xg))

            for a in range(nloc):

                Ia     = int(dof_indices[a])
                gradNa = invjacg @ gN[g, a]   # ∇_x N_a = J^{-T} ∇_ξ N_a

                # F_a += w_g · f(x_g) · N_a · |det J|
                F[Ia] += wg * f_g * N[g, a] * detg

                for b in range(nloc):

                    Ib     = int(dof_indices[b])
                    gradNb = invjacg @ gN[g, b]

                    # K_ab += w_g · κ(u_h, x_g) · (∇N_a · ∇N_b) · |det J|
                    K[Ia, Ib] += wg * kappa_g * float(np.dot(gradNa, gradNb)) * detg

    return K, F