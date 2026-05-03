# =============================================================================
# imex_solver.py — Un pas de temps IMEX pour l'équation de Fisher-KPP
# =============================================================================
#
# L'équation à résoudre est :
#
#   ∂u/∂t - ∇·(κ(u,x)∇u) = r·u·(1 - u/K(x))
#     terme de diffusion       terme de réaction
#
# POURQUOI LE SCHÉMA IMEX ?
# --------------------------
# On a deux termes de natures mathématiques opposées :
#
#   Diffusion : opérateur LINÉAIRE en u, mais RAIDE.
#     → Traitement explicite interdit : imposerait Δt < h²/(2κ) ≈ 0.23 an.
#     → Doit être traité IMPLICITEMENT pour la stabilité.
#
#   Réaction  : terme NON LINÉAIRE (contient u²), mais DOUX (varie lentement).
#     → Traitement implicite forcerait à résoudre un système non linéaire.
#     → Peut être traité EXPLICITEMENT sans risque d'instabilité si Δt·r < 1.
#
# Le schéma IMEX traite chaque terme selon sa nature :
#
#   (M + Δt·K(uⁿ)) · u^{n+1} = M·uⁿ + Δt · r·uⁿ·(1 - uⁿ/K) · M_lump
#    \_______diffusion implicite_/   \________réaction explicite_________/
#
# K(uⁿ) est la matrice de rigidité assemblée avec κ évalué en uⁿ (connu).
# Le système reste donc LINÉAIRE en u^{n+1} → résolution directe possible.
#
# CONDITION DE STABILITÉ pour la réaction explicite : Δt · r < 1
# Avec r = 1.0 an⁻¹ et Δt = 0.1 an : 0.1 < 1 ✓
# =============================================================================

import numpy as np
from stiffness_non_linear import assemble_stiffness_and_rhs
from dirichlet import theta_step


def imex_step(U_old, problem, dt, theta=1.0):
    """
    Effectue un pas de temps IMEX pour l'équation de Fisher-KPP.

    Paramètres
    ----------
    U_old   : solution nodale à l'instant tⁿ (num_dofs,)
    problem : dictionnaire renvoyé par build_problem()
    dt      : pas de temps [années]
    theta   : paramètre du schéma theta
              1.0 → Euler implicite (stable, ordre 1)
              0.5 → Crank-Nicolson  (stable, ordre 2)

    Retour
    ------
    U_new : solution nodale à l'instant t^{n+1} (num_dofs,)
    """

    M        = problem["M"]          
    M_lump   = problem["M_lump"]     
    K_nodal  = problem["K_nodal"]    
    R_GROWTH = problem["R_GROWTH"]   
    dir_dofs = problem["dir_dofs"]   
    dir_vals = problem["dir_vals"]   

    # =========================================================================
    # ÉTAPE 1 : Matrice de rigidité K avec κ(uⁿ, x)
    # =========================================================================
    # κ est évalué en uⁿ, ce qui maintient K linéaire en u^{n+1}.
    # Terme source nul (0.0) car la réaction logistique est gérée séparément.
    K_lil, F0 = assemble_stiffness_and_rhs(
        problem["elemTags"],
        problem["elemNodeTags"],
        problem["jac"],
        problem["det"],
        problem["coords"],
        problem["w"],
        problem["N"],
        problem["gN"],
        U_old,
        problem["kappa_fun"],
        lambda x: 0.0,
        problem["tag_to_dof"]
    )
    K_mat = K_lil.tocsr()

    # f(uⁿ) = r · uⁿ · (1 - uⁿ/K(x))
    # L'utilisation de la masse lumpée (M_lump) évite un produit matrice-vecteur complet M·f.
    # Le garde-fou max(U_old, 0) évite une croissance artificielle sur des artefacts négatifs.
    U_pos   = np.maximum(U_old, 0.0)
    f_react = R_GROWTH * U_pos * (1.0 - U_pos / K_nodal)

    # Second membre total = contribution diffusion (F0) + réaction
    F_total = F0 + f_react * M_lump

    # Résout : (M + θ·Δt·K)·u^{n+1} = (M - (1-θ)·Δt·K)·uⁿ + Δt·F
    # F_total est passé pour F_n et F_{n+1} car la réaction est 100% évaluée en uⁿ.
    U_new = theta_step(
        M, K_mat,
        F_total, F_total,
        U_old,
        dt=dt,
        theta=theta,
        dirichlet_dofs=dir_dofs,
        dir_vals_np1=dir_vals
    )

    # garde-fous physiques
    U_new = np.maximum(U_new, 0.0)
    U_new[dir_dofs] = dir_vals

    return U_new