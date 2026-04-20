# =============================================================================
# imex_solver.py — Un pas de temps IMEX pour l'équation de Fisher-KPP
# =============================================================================
#
# L'équation à résoudre est :
#
#     ∂u/∂t - ∇·(κ(u,x)∇u) = r·u·(1 - u/K(x))
#       terme de diffusion       terme de réaction
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

    # Lecture des données nécessaires depuis le dictionnaire du problème
    M        = problem["M"]          # matrice de masse (assemblée une seule fois)
    M_lump   = problem["M_lump"]     # masse lumpée pour la réaction (vecteur)
    K_nodal  = problem["K_nodal"]    # capacité de charge locale aux nœuds [ind/km²]
    R_GROWTH = problem["R_GROWTH"]   # taux de croissance r [an⁻¹]
    dir_dofs = problem["dir_dofs"]   # indices des DDLs de Dirichlet (bord de mer)
    dir_vals = problem["dir_vals"]   # valeurs imposées (u = 0 sur la côte)

    # =========================================================================
    # ÉTAPE 1 — Assemblage de la matrice de rigidité K avec κ(uⁿ, x)
    # =========================================================================
    #
    # On passe U_old (= uⁿ, déjà connu) à assemble_stiffness_and_rhs.
    # À l'intérieur, κ sera évalué en uⁿ à chaque point de Gauss.
    # Puisque uⁿ est un nombre fixé au moment du calcul, κ(uⁿ, x) se comporte
    # comme une simple fonction d'espace : K reste linéaire en u^{n+1}.
    #
    # Le terme source volumique supplémentaire est nul (lambda x: 0.0) :
    # toute la source vient de la réaction logistique traitée séparément.

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

    # =========================================================================
    # ÉTAPE 2 — Terme de réaction logistique (explicite en uⁿ)
    # =========================================================================
    #
    # f(uⁿ) = r · uⁿ · (1 - uⁿ/K(x))
    #
    # Interprétation physique :
    #   - si u << K : f ≈ r·u   → croissance presque exponentielle
    #   - si u → K  : f → 0     → croissance s'annule à saturation
    #   - si u > K  : f < 0     → la population décroît (surpopulation)
    #
    # On utilise la masse lumpée M_lump plutôt que la matrice M complète :
    # M_lump[i] = somme de la ligne i de M.
    # Cela remplace le produit matrice-vecteur M·f par une simple
    # multiplication terme à terme M_lump * f_react, bien plus rapide.
    #
    # Le garde-fou max(U_old, 0) empêche une densité légèrement négative
    # (artefact numérique possible) de produire une croissance artificielle.

    U_pos   = np.maximum(U_old, 0.0)
    f_react = R_GROWTH * U_pos * (1.0 - U_pos / K_nodal)

    # Second membre total = contribution diffusion (F0) + réaction
    F_total = F0 + f_react * M_lump

    # =========================================================================
    # ÉTAPE 3 — Résolution implicite du système linéaire
    # =========================================================================
    #
    # theta_step résout : (M + θ·Δt·K)·u^{n+1} = (M - (1-θ)·Δt·K)·uⁿ + Δt·F
    #
    # On passe F_total comme F_n ET F_{n+1} car la réaction est entièrement
    # évaluée en uⁿ : le terme source ne change pas entre tⁿ et t^{n+1}.
    # Passer le même vecteur deux fois est donc mathématiquement correct.
    #
    # Les conditions de Dirichlet (u=0 sur la côte) sont imposées à l'intérieur
    # de theta_step par réduction du système linéaire.

    U_new = theta_step(
        M, K_mat,
        F_total, F_total,
        U_old,
        dt=dt,
        theta=theta,
        dirichlet_dofs=dir_dofs,
        dir_vals_np1=dir_vals
    )

    # =========================================================================
    # ÉTAPE 4 — Garde-fous physiques
    # =========================================================================
    #
    # La résolution numérique peut produire de très légères valeurs négatives
    # près des bords ou dans les zones à faible densité. Une densité de
    # population ne peut pas être négative : on la force à zéro.
    #
    # On réimpose aussi Dirichlet par sécurité, au cas où la résolution
    # aurait légèrement perturbé les valeurs aux nœuds imposés.

    U_new = np.maximum(U_new, 0.0)
    U_new[dir_dofs] = dir_vals

    return U_new