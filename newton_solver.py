# =============================================================================
# newton_solver.py — Solveur Newton-Raphson pour l'équation de Fisher-KPP
# =============================================================================
#
# À chaque pas de temps, on cherche u^{n+1} tel que R(u^{n+1}) = 0 avec :
#
#   R(U) = (M/Δt)(U - U^n)  +  ∫ κ(u,x) ∇u·∇Na dΩ  -  ∫ r·u·(1-u/K)·Na dΩ
#           terme temporel     diffusion implicite     réaction implicite
#
# Contrairement à l'IMEX, aucun terme n'est gelé à uⁿ : tout est évalué
# en u^{n+1} (l'inconnue). Le système est donc non linéaire et on le résout
# par Newton-Raphson.
#
# PRINCIPE DE NEWTON-RAPHSON :
# À partir d'un guess initial U⁰ (= U^n en général), on itère :
#
#   J(Uᵏ) · δU = -R(Uᵏ)      (résolution d'un système linéaire)
#   U^{k+1} = Uᵏ + δU         (mise à jour)
#
# jusqu'à ce que ‖R(Uᵏ)‖ < tolérance (convergence quadratique près de la solution).
#
# CE FICHIER CONTIENT 4 FONCTIONS :
#   preprocess_newton_data  → pré-calcul one-shot des données FEM
#   assemble_residual       → calcul de R(U)
#   assemble_jacobian       → calcul de J(U) = ∂R/∂U
#   newton_solver           → boucle Newton qui orchestre les 3 précédentes
# =============================================================================

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import numpy as np


# =============================================================================
# FONCTION 1 — Prétraitement : données FEM indépendantes de u et du temps
# =============================================================================

def preprocess_newton_data(elemTags, conn, jac, det, xphys, w, N, gN, tag_to_dof, K_nodal=None):
    """
    Pré-calcule une fois pour toutes les données FEM qui ne changent pas
    au cours de la simulation (ni avec le temps, ni avec les itérations Newton).

    L'intérêt est de ne pas recalculer les inversions de jacobiens et les
    gradients physiques à chaque itération Newton de chaque pas de temps.
    Ces opérations sont coûteuses (np.linalg.inv sur ne*ngp matrices 3*3)
    et leur résultat est identique à chaque appel puisqu'il ne dépend que
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

    Retour
    ------
    data : dict contenant toutes les données pré-calculées
    """

    # --- Dimensions ---
    ne  = len(elemTags)
    ngp = len(w)

    # --- Mise en forme de la connectivité ---
    # Gmsh peut renvoyer la connectivité aplatie (ne*nloc,) ou déjà en (ne, nloc).
    conn = np.asarray(conn, dtype=np.int64)
    if conn.ndim == 1:
        nloc = conn.size // ne
        conn = conn.reshape(ne, nloc)
    else:
        nloc = conn.shape[1]

    # --- Mise en forme des autres tableaux Gmsh ---
    det   = np.asarray(det,   dtype=np.float64).reshape(ne, ngp)
    xphys = np.asarray(xphys, dtype=np.float64).reshape(ne, ngp, 3)
    jac   = np.asarray(jac,   dtype=np.float64).reshape(ne, ngp, 3, 3)
    N     = np.asarray(N,     dtype=np.float64).reshape(ngp, nloc)
    gN    = np.asarray(gN,    dtype=np.float64).reshape(ngp, nloc, 3)
    w     = np.asarray(w,     dtype=np.float64)

    # --- Conversion tags → indices DDL compacts ---
    # dof_indices[e, a] = indice global du DDL local a de l'élément e
    dof_indices = tag_to_dof[conn]   # shape (ne, nloc)

    # --- Restriction de K_nodal aux éléments ---
    # K_elem[e, a] = valeur de K au nœud local a de l'élément e.
    # Cela permet d'interpoler K au point de Gauss par K_g = Σ_a K_elem[e,a]·N[g,a],
    # exactement comme on interpole u_h. On évite ainsi d'appeler une fonction
    # K(x) à chaque point de Gauss (ce qui serait plus lent).
    K_elem = K_nodal[dof_indices] if K_nodal is not None else None

    # --- Inversion des jacobiens (une seule fois) ---
    # invjac[e, g] = J^{-1} au point de Gauss g de l'élément e.
    # Servira à calculer les gradients physiques : ∇_x N = J^{-T} ∇_ξ N.
    invjac = np.linalg.inv(jac)   # shape (ne, ngp, 3, 3)

    # --- Gradients physiques des fonctions de forme (une seule fois) ---
    # gradN_phys[e, g, a, :] = ∇_x N_a au point de Gauss g de l'élément e.
    # Ces gradients ne dépendent que de la géométrie → calculés une seule fois.
    gradN_phys = np.zeros((ne, ngp, nloc, 3), dtype=np.float64)
    for e in range(ne):
        for g in range(ngp):
            for a in range(nloc):
                gradN_phys[e, g, a, :] = invjac[e, g] @ gN[g, a]

    nn = int(np.max(dof_indices) + 1)   # nombre total de DDLs

    return {
        "ne": ne, "ngp": ngp, "nloc": nloc, "nn": nn,
        "conn": conn, "dof_indices": dof_indices,
        "det": det, "xphys": xphys, "jac": jac, "invjac": invjac,
        "w": w, "N": N, "gN": gN,
        "gradN_phys": gradN_phys,
        "K_elem": K_elem,
    }


# =============================================================================
# FONCTION 2 — Assemblage du résidu R(U)
# =============================================================================

def assemble_residual(U, U_old, M, dt, newton_data, kappa_fun, r_growth, dirichlet_dofs=None, dirichlet_vals=None):
    """
    Calcule le résidu R(U) du problème non linéaire à résoudre.

    On cherche U^{n+1} tel que R(U^{n+1}) = 0, avec :

        R(U) = R1 + R2 - R3

    R1 (terme temporel)  : R1 = (M/Δt)(U - U^n)
    R2 (diffusion)       : R2_a = ∫ κ(u,x) ∇u · ∇Na dΩ
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
    r_growth      : taux de croissance r [an⁻¹]
    dirichlet_dofs: indices des DDLs imposés
    dirichlet_vals: valeurs imposées

    Retour
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

    # R1 : terme temporel — simple produit matrice-vecteur sparse
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

            # Reconstruction de u_h et ∇u_h au point de Gauss
            # u_h = Σ_b U_b · N_b(ξ_g)  et  ∇u_h = Σ_b U_b · ∇_x N_b
            u_g      = float(np.dot(Ue, N[g, :]))
            grad_u_g = np.einsum('b,bd->d', Ue, gradN_phys[e, g])

            # Reconstruction de K_g au point de Gauss par interpolation
            # K n'est pas une fonction analytique mais un champ nodal :
            # K_g = Σ_b K_elem[e,b] · N_b(ξ_g)
            K_g = float(np.dot(K_elem[e], N[g, :]))

            kappa_g = float(kappa_fun(u_g, xg))

            for a in range(nloc):
                Ia     = int(idx[a])
                Na     = N[g, a]
                gradNa = gradN_phys[e, g, a]

                # R2 : contribution de la diffusion au résidu
                # R2_a += w_g · |det J| · κ(u,x) · (∇u · ∇Na)
                R2[Ia] += wg * detg * kappa_g * np.dot(grad_u_g, gradNa)

                # R3 : contribution de la réaction au résidu
                # R3_a += w_g · |det J| · r·u·(1 - u/K) · Na
                R3[Ia] += wg * detg * r_growth * u_g * (1.0 - u_g / K_g) * Na

    R = R1 + R2 - R3

    # Conditions de Dirichlet : on remplace R_i par (U_i - U_D_i)
    # Ainsi Newton cherchera U_i = U_D_i à convergence.
    if dirichlet_dofs is not None and dirichlet_vals is not None:
        R[dirichlet_dofs] = U[dirichlet_dofs] - dirichlet_vals

    return R


# =============================================================================
# FONCTION 3 — Assemblage de la jacobienne J(U) = ∂R/∂U
# =============================================================================

def assemble_jacobian(U, M, dt, newton_data, kappa_fun, dkappa_du, r_growth, dirichlet_dofs=None):
    """
    Calcule la jacobienne J(U) = ∂R/∂U du résidu.

    Newton résout J(Uᵏ)·δU = -R(Uᵏ). Pour cela il faut J, la matrice des
    dérivées partielles de R par rapport aux inconnues nodales.

        J = J1 + J2 - J3

    J1 (terme temporel)  : J1 = M/Δt  (immédiat, R1 est linéaire en U)

    J2 (diffusion)       : deux contributions car κ dépend de u :
        J2[a,b] = ∫ κ(u,x)·∇Nb·∇Na dΩ          (variation de ∇u)
                + ∫ (∂κ/∂u)·Nb·(∇u·∇Na) dΩ     (variation de κ(u))

    J3 (réaction)        : f'(u) = r·(1 - 2u/K)
        J3[a,b] = ∫ f'(u)·Nb·Na dΩ

    Pour les DDLs de Dirichlet, on impose J_ii = 1, J_ij = 0 (i≠j)
    ET on annule la colonne i dans les autres lignes, pour que la
    correction δU_i soit nulle (la valeur est déjà imposée dans R).

    Paramètres
    ----------
    U             : solution courante
    M             : matrice de masse (sparse)
    dt            : pas de temps
    newton_data   : dict renvoyé par preprocess_newton_data()
    kappa_fun     : κ(u, x)
    dkappa_du     : ∂κ/∂u(u, x)
    r_growth      : taux de croissance r
    dirichlet_dofs: indices des DDLs imposés

    Retour
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

    # J1 est immédiat : R1 = (M/dt)(U - U_old) est linéaire en U
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

            # Dérivée de f(u) = r·u·(1 - u/K) par rapport à u :
            # f'(u) = r·(1 - 2u/K)
            # Note : le facteur 2 vient du terme en u² lors de la dérivation.
            df_du = r_growth * (1.0 - 2.0 * u_g / K_g)

            for b in range(nloc):
                Nb     = N[g, b]
                Ib     = int(idx[b])
                gradNb = gradN_phys[e, g, b]

                for a in range(nloc):
                    Na     = N[g, a]
                    Ia     = int(idx[a])
                    gradNa = gradN_phys[e, g, a]

                    # J2 : jacobienne de la diffusion — deux termes (règle du produit)
                    #
                    # 1er terme : variation de ∇u quand U_b change
                    #   ∂/∂U_b [κ(u) ∇u · ∇Na] = κ(u) · ∇Nb · ∇Na
                    term_grad = kappa_g * np.dot(gradNb, gradNa)

                    # 2e terme : variation de κ(u) quand U_b change (car κ dépend de u)
                    #   ∂κ/∂u · (∂u/∂U_b) · (∇u · ∇Na) = (∂κ/∂u) · Nb · (∇u · ∇Na)
                    term_kappa = dkappa_g * Nb * np.dot(grad_u_g, gradNa)

                    J2[Ia, Ib] += (term_grad + term_kappa) * wg * detg

                    # J3 : jacobienne de la réaction
                    # ∂/∂U_b [f(u)·Na] = f'(u)·Nb·Na
                    J3[Ia, Ib] += df_du * Nb * Na * wg * detg

    J = (J1 + J2 - J3).tolil()

    # --- Conditions de Dirichlet dans la jacobienne ---
    # Pour chaque DDL imposé i :
    #   - ligne i → identité : [0 ... 1 ... 0]  (δU_i = -R_i/1 = 0 car R_i=U_i-U_D_i)
    #   - colonne i → zéro dans toutes les autres lignes
    #     (sinon le DDL imposé "polluerait" les équations des DDLs libres voisins)
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


# =============================================================================
# FONCTION 4 — Boucle Newton-Raphson
# =============================================================================

def newton_solver(U_init, U_old, M, dt, newton_data, kappa_fun, dkappa_du, r_growth, dirichlet_dofs=None, dirichlet_vals=None, tol=1e-5, max_iter=20):
    """
    Résout R(U^{n+1}) = 0 par la méthode de Newton-Raphson.

    À chaque itération k, on linéarise R autour de Uᵏ :
        R(U^{k+1}) ≈ R(Uᵏ) + J(Uᵏ)·δU = 0
    ce qui donne le système linéaire à résoudre :
        J(Uᵏ)·δU = -R(Uᵏ)
    puis la mise à jour :
        U^{k+1} = Uᵏ + δU

    Newton converge quadratiquement près de la solution : si ‖δU‖ ≈ ε,
    l'itération suivante donne ‖δU‖ ≈ ε². C'est bien plus rapide que
    les méthodes du premier ordre (Picard, point fixe).

    Deux critères d'arrêt :
      - ‖R(Uᵏ)‖ < tol  : le résidu est négligeable (critère physique)
      - ‖δU‖ < tol      : la correction est négligeable (critère numérique)

    Paramètres
    ----------
    U_init        : guess initial (en général U_old = u^n)
    U_old         : solution au pas précédent u^n
    M             : matrice de masse (sparse)
    dt            : pas de temps
    newton_data   : dict renvoyé par preprocess_newton_data()
    kappa_fun     : κ(u, x)
    dkappa_du     : ∂κ/∂u(u, x)
    r_growth      : taux de croissance r
    dirichlet_dofs: indices des DDLs imposés
    dirichlet_vals: valeurs imposées
    tol           : tolérance de convergence (défaut : 1e-5)
    max_iter      : nombre maximal d'itérations (défaut : 20)

    Retour
    ------
    U : solution convergée u^{n+1}
    """

    U = U_init.copy()   # on ne modifie pas U_init directement

    for k in range(max_iter):

        # Assemblage du résidu à l'itération courante
        R = assemble_residual(
            U, U_old, M, dt,
            newton_data, kappa_fun, r_growth,
            dirichlet_dofs, dirichlet_vals
        )

        # Critère d'arrêt sur le résidu
        if np.linalg.norm(R) < tol:
            break

        # Assemblage de la jacobienne à l'itération courante
        J = assemble_jacobian(
            U, M, dt,
            newton_data, kappa_fun, dkappa_du, r_growth,
            dirichlet_dofs
        )

        # Résolution du système linéaire J·δU = -R
        deltaU = spsolve(J, -R)

        # Mise à jour de la solution
        U += deltaU

        # Critère d'arrêt secondaire : la correction est devenue négligeable
        if np.linalg.norm(deltaU) < tol:
            break

    return U