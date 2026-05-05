# imex_solver.py
# solveur imex - fait avancer la solution en temps pas à pas 
# diffusion est traitée implicitement et la réaction (terme logistque de croissance ) est traitée explicitement
# Equation globale : du/dt - div (kappa(u,x) grad(u)) = ru (1 - u/K(x))
# Schéma IMEX associé : (M + delta(t) K(u^n,x)) . u^(n+1) = Mu^n + delta(t) ru^n (1 - u^n/K) . M_lump
# theta-step : (M + theta delta(t) K) u^(n+1) = (M - (1-theta) delta(t) K) u^n + delta(t) F_total

import numpy as np
from stiffness_non_linear import assemble_stiffness_and_rhs
from dirichlet import theta_step


def imex_step(U_old, problem, dt, theta=1.0):
    """
    Effectue un pas de temps IMEX pour l'équation de Fisher-KPP.

    Paramètres
    ----------
    U_old   : solution nodale à l'instant t^n
    problem : dictionnaire renvoyé par build_problem()
    dt      : pas de temps [années]
    theta   : paramètre du schéma theta
              1.0 : Euler implicite (stable, ordre 1)
              0.5 : Crank-Nicolson  (stable, ordre 2)

    Return
    ------
    U_new : solution nodale à l'instant t^(n+1)
    """

    # Lecture des données nécessaires depuis le dictionnaire du problème
    M = problem["M"]          # matrice de masse (assemblée une seule fois)
    M_lump = problem["M_lump"]     # masse lumpée pour la réaction (vecteur)
    K_nodal = problem["K_nodal"]    # capacité de charge locale aux nœuds [ind/km²]
    R_GROWTH = problem["R_GROWTH"]   # taux de croissance r [an⁻¹]
    dir_dofs = problem["dir_dofs"]   # indices des DDLs de Dirichlet (bord de mer)
    dir_vals = problem["dir_vals"]   # valeurs imposées (u = 0 sur la côte)

    # assemblage du terme de diffusion (implicite)
    K_lil, F0 = assemble_stiffness_and_rhs(
        problem["elemTags"],
        problem["elemNodeTags"],
        problem["jac"],
        problem["det"],
        problem["coords"],
        problem["w"],
        problem["N"],
        problem["gN"],
        U_old,  # kappa est évalué avec u^n
        problem["kappa_fun"],
        lambda x: 0.0,
        problem["tag_to_dof"]
    )
    K_mat = K_lil.tocsr()

    # terme de réaction explicite
    U_pos   = np.maximum(U_old, 0.0)
    f_react = R_GROWTH * U_pos * (1.0 - U_pos / K_nodal)
    F_total = F0 + f_react * M_lump

    #theta step
    U_new = theta_step(
        M, K_mat,
        F_total, F_total,
        U_old,
        dt=dt,
        theta=theta,
        dirichlet_dofs=dir_dofs,
        dir_vals_np1=dir_vals
    )

    U_new = np.maximum(U_new, 0.0)  #u>=0, densité ne peut pas etre -
    U_new[dir_dofs] = dir_vals  #u = 0 sur la côte

    return U_new