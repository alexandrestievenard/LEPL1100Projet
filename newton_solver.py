from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import numpy as np


def preprocess_newton_data(elemTags, conn, jac, det, xphys, w, N, gN, tag_to_dof, K_nodal=None):
    """
    Pré-calcule une fois pour toutes les données FEM qui ne change pas
    au cours de la simulation (ni avec le temps, ni avec les itérations Newton).

    L'intérêt est de ne pas recalculer les inversions de jacobiens et les
    gradients physiques à chaque itération Newton de chaque pas de temps.
    Ces opérations sont coûteuses et leur résultat est identique à chaque appel puisqu'il ne dépend que
    de la géométrie du maillage, qui est fixe.

    Paramètres
    ----------
    elemTags   : tags des éléments (ne,)
    conn       : connectivité aplatie ou (ne, nloc) en tags Gmsh
    jac        : jacobiens aplatis ou (ne, ngp, 3, 3)
    det        : déterminants aplatis (ne*ngp,)
    xphys      : coordonnées physiques des pts de Gauss aplaties (ne*ngp*3,)
    w          : poids de quadrature (ngp,)
    N          : valeurs des fonctions de forme (ngp*nloc,)
    gN         : gradients en coordonnées de référence (ngp*nloc*3,)
    tag_to_dof : correspondance tag Gmsh → indice DDL compact
    K_nodal    : capacité de charge aux nœuds (nn,) — optionnel

    Return
    ------
    data : dict contenant toutes les données pré-calculées
    """
    ne  = len(elemTags)
    ngp = len(w)

    conn = np.asarray(conn, dtype=np.int64)
    if conn.ndim == 1:
        nloc = conn.size // ne
        conn = conn.reshape(ne, nloc)
    else:
        nloc = conn.shape[1]

    det = np.asarray(det, dtype=np.float64).reshape(ne, ngp)
    xphys = np.asarray(xphys, dtype=np.float64).reshape(ne, ngp, 3)
    jac = np.asarray(jac,dtype=np.float64).reshape(ne, ngp, 3, 3)
    N = np.asarray(N, dtype=np.float64).reshape(ngp, nloc)
    gN = np.asarray(gN, dtype=np.float64).reshape(ngp, nloc, 3)
    w = np.asarray(w, dtype=np.float64)

    dof_indices = tag_to_dof[conn] 
    K_elem = K_nodal[dof_indices] if K_nodal is not None else None
    invjac = np.linalg.inv(jac)

    gradN_phys = np.zeros((ne, ngp, nloc, 3), dtype=np.float64)
    for e in range(ne):
        for g in range(ngp):
            for a in range(nloc):
                gradN_phys[e, g, a, :] = invjac[e, g] @ gN[g, a]

    nn = int(np.max(dof_indices) + 1) 

    return {
        "ne": ne, "ngp": ngp, "nloc": nloc, "nn": nn,
        "conn": conn, "dof_indices": dof_indices,
        "det": det, "xphys": xphys, "jac": jac, "invjac": invjac,
        "w": w, "N": N, "gN": gN,
        "gradN_phys": gradN_phys,
        "K_elem": K_elem,
    }


def assemble_residual(U, U_old, M, dt, newton_data, kappa_fun, r_growth, dirichlet_dofs=None, dirichlet_vals=None):
    """
    Calcule le résidu R(U) du problème non linéaire à résoudre.

    On cherche U^{n+1} tel que R(U^{n+1}) = 0, avec :

        R(U) = R1 + R2 - R3

    R1 (terme temporel)  : R1 = (M/dleta(t))(U - U^n)
    R2 (diffusion)       : R2_a = ∫ kappa(u,x) grad(u) · grad(Na) dΩ
    R3 (réaction)        : R3_a = ∫ r·u·(1 - u/K) · Na dΩ

    Aux DDLs de Dirichlet, on remplace R_i par (U_i - U_D_i)
    pour forcer la solution à respecter la valeur imposée.

    Paramètres
    ----------
    U             : solution courante u^{n+1,(k)} à l'itération Newton k
    U_old         : solution au pas précédent u^n
    M             : matrice de masse (sparse)
    dt            : pas de temps
    newton_data   : dict renvoyé par preprocess_newton_data()
    kappa_fun     : κ(u, x)
    r_growth      : taux de croissance r [1/an]
    dirichlet_dofs: indices des DDLs imposés
    dirichlet_vals: valeurs imposées

    Return
    ------
    R : vecteur résidu (nn,)
    """

    ne  = newton_data["ne"]
    ngp = newton_data["ngp"]
    nloc = newton_data["nloc"]
    nn   = newton_data["nn"]

    dof_indices = newton_data["dof_indices"]
    det         = newton_data["det"]
    xphys       = newton_data["xphys"]
    w           = newton_data["w"]
    N           = newton_data["N"]
    gradN_phys  = newton_data["gradN_phys"]
    K_elem      = newton_data["K_elem"]

    if K_elem is None:
        raise ValueError("K_elem manquant dans newton_data. Fournir K_nodal à preprocess_newton_data.")

    R1 = (M @ (U - U_old)) / dt
    R2 = np.zeros(nn)
    R3 = np.zeros(nn)

    for e in range(ne):
        idx = dof_indices[e]   # indices globaux des DDLs de l'élément e
        Ue  = U[idx]           # valeurs nodales de u^{n+1} sur l'élément e

        for g in range(ngp):
            xg   = xphys[e, g]
            wg   = w[g]
            detg = det[e, g]

            u_g      = float(np.dot(Ue, N[g, :]))
            grad_u_g = np.einsum('b,bd->d', Ue, gradN_phys[e, g])

            K_g = float(np.dot(K_elem[e], N[g, :]))

            kappa_g = float(kappa_fun(u_g, xg))

            for a in range(nloc):
                Ia     = int(idx[a])
                Na     = N[g, a]
                gradNa = gradN_phys[e, g, a]
                
                R2[Ia] += wg * detg * kappa_g * np.dot(grad_u_g, gradNa)
                R3[Ia] += wg * detg * r_growth * u_g * (1.0 - u_g / K_g) * Na

    R = R1 + R2 - R3


    if dirichlet_dofs is not None and dirichlet_vals is not None:
        R[dirichlet_dofs] = U[dirichlet_dofs] - dirichlet_vals

    return R

def assemble_jacobian(U, M, dt, newton_data, kappa_fun, dkappa_du, r_growth, dirichlet_dofs=None):
    """
    Calcule la jacobienne J(U) = dR/dU du résidu.

    Newton résout J(Uᵏ)·delta(U) = -R(U^k). Pour cela il faut J, la matrice des
    dérivées partielles de R par rapport aux inconnues nodales.

        J = J1 + J2 - J3

    J1 (terme temporel)  : J1 = M/delta(t)  (immédiat, R1 est linéaire en U)

    J2 (diffusion)       : deux contributions car κ dépend de u :
        J2[a,b] = ∫ kappa(u,x)·grad(Nb)·grad(Na))         (variation de delta(u)
                + ∫ (dkappa/du)·Nb·(grad(u)·grad(Na))     (variation de kappa(u))

    J3 (réaction)        : f'(u) = r·(1 - 2u/K)
        J3[a,b] = ∫ f'(u)·Nb·Na domega

    Pour les DDLs de Dirichlet, on impose J_ii = 1, J_ij = 0 (i!=j)
    ET on annule la colonne i dans les autres lignes, pour que la
    correction δU_i soit nulle (la valeur est déjà imposée dans R).

    Paramètres
    ----------
    U             : solution courante
    M             : matrice de masse (sparse)
    dt            : pas de temps
    newton_data   : dict renvoyé par preprocess_newton_data()
    kappa_fun     : kappa(u, x)
    dkappa_du     : dkappa/du(u, x)
    r_growth      : taux de croissance r
    dirichlet_dofs: indices des DDLs imposés

    Return
    ------
    J : jacobienne globale (sparse CSR, nn * nn)
    """

    ne   = newton_data["ne"]
    ngp  = newton_data["ngp"]
    nloc = newton_data["nloc"]
    nn   = newton_data["nn"]

    dof_indices = newton_data["dof_indices"]
    det         = newton_data["det"]
    xphys       = newton_data["xphys"]
    w           = newton_data["w"]
    N           = newton_data["N"]
    gradN_phys  = newton_data["gradN_phys"]
    K_elem      = newton_data["K_elem"]

    if K_elem is None:
        raise ValueError("K_elem manquant dans newton_data. Fournir K_nodal à preprocess_newton_data.")

    J1 = (M / dt).tolil()
    J2 = lil_matrix((nn, nn), dtype=np.float64)
    J3 = lil_matrix((nn, nn), dtype=np.float64)

    for e in range(ne):
        idx = dof_indices[e]
        Ue  = U[idx]

        for g in range(ngp):
            xg   = xphys[e, g]
            wg   = w[g]
            detg = det[e, g]

            u_g      = float(np.dot(Ue, N[g, :]))
            grad_u_g = np.einsum('b,bd->d', Ue, gradN_phys[e, g])
            K_g      = float(np.dot(K_elem[e], N[g, :]))

            kappa_g  = float(kappa_fun(u_g, xg))
            dkappa_g = float(dkappa_du(u_g, xg))
            
            df_du = r_growth * (1.0 - 2.0 * u_g / K_g)

            for b in range(nloc):
                Nb     = N[g, b]
                Ib     = int(idx[b])
                gradNb = gradN_phys[e, g, b]

                for a in range(nloc):
                    Na     = N[g, a]
                    Ia     = int(idx[a])
                    gradNa = gradN_phys[e, g, a]
                    term_grad = kappa_g * np.dot(gradNb, gradNa)
                    term_kappa = dkappa_g * Nb * np.dot(grad_u_g, gradNa)

                    J2[Ia, Ib] += (term_grad + term_kappa) * wg * detg

                    J3[Ia, Ib] += df_du * Nb * Na * wg * detg

    J = (J1 + J2 - J3).tolil()

    if dirichlet_dofs is not None:
        dir_set = set(int(i) for i in dirichlet_dofs)

        for i in dirichlet_dofs:
            J.rows[i] = [i]
            J.data[i] = [1.0]

        for row in range(J.shape[0]):
            if row in dir_set:
                continue
            new_cols = [c for c in J.rows[row] if c not in dir_set]
            new_vals = [v for c, v in zip(J.rows[row], J.data[row]) if c not in dir_set]
            J.rows[row] = new_cols
            J.data[row] = new_vals

    return J.tocsr()

def newton_solver(U_init, U_old, M, dt, newton_data, kappa_fun, dkappa_du, r_growth, dirichlet_dofs=None, dirichlet_vals=None, tol=1e-5, max_iter=20):
    """
    Résout R(U^{n+1}) = 0 par la méthode de Newton-Raphson.

    À chaque itération k, on linéarise R autour de Uᵏ :
        R(U^{k+1}) ~ R(U^k) + J(U^k)·delat(U) = 0
    ce qui donne le système linéaire à résoudre :
        J(U^k)·delta(U) = -R(U^k)
    puis la mise à jour :
        U^{k+1} = U^k + delta(U)
    Newton converge quadratiquement près de la solution : si ‖δU‖ ~ epsilone,
    l'itération suivante donne ‖delat(U)‖ ~ epsilon². C'est bien plus rapide que
    les méthodes du premier ordre (Picard, point fixe).
    Deux critères d'arrêt :
      - ‖R(U^k)‖ < tol : le résidu est négligeable (critère physique)
      - ‖delta(U)‖ < tol : la correction est négligeable (critère numérique)

    Paramètres
    ----------
    U_init        : guess initial (en général U_old = u^n)
    U_old         : solution au pas précédent u^n
    M             : matrice de masse (sparse)
    dt            : pas de temps
    newton_data   : dict renvoyé par preprocess_newton_data()
    kappa_fun     : kappa(u, x)
    dkappa_du     : dkappa/du(u, x)
    r_growth      : taux de croissance r
    dirichlet_dofs: indices des DDLs imposés
    dirichlet_vals: valeurs imposées
    tol           : tolérance de convergence (défaut : 1e-5)
    max_iter      : nombre maximal d'itérations (défaut : 20)

    Return
    ------
    U : solution convergée u^{n+1}
    """

    U = U_init.copy()

    for k in range(max_iter):

        R = assemble_residual(
            U, U_old, M, dt,
            newton_data, kappa_fun, r_growth,
            dirichlet_dofs, dirichlet_vals
        )

        if np.linalg.norm(R) < tol:
            break
        
        J = assemble_jacobian(
            U, M, dt,
            newton_data, kappa_fun, dkappa_du, r_growth,
            dirichlet_dofs
        )

        deltaU = spsolve(J, -R)
        
        U += deltaU

        if np.linalg.norm(deltaU) < tol:
            break

    return U   