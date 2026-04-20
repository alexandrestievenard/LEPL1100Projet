# =============================================================================
# dirichlet.py
# =============================================================================
# Ce fichier contient deux fonctionnalités distinctes :
#
#   1. Imposition des conditions de Dirichlet par réduction du système
#      On a K·U = F et certains DDLs ont des valeurs imposées.
#      On réduit le système aux seuls DDLs libres avant de résoudre.
#
#   2. Avance en temps par le schéma θ
#      Discrétisation de M·∂u/∂t + K·u = F(t) :
#
#      (M + θΔt K)·u^{n+1} = (M - (1-θ)Δt K)·u^n + Δt·(θ F^{n+1} + (1-θ) F^n)
#
#      θ = 1   → Euler implicite   (stable, ordre 1 en temps)
#      θ = 0.5 → Crank-Nicolson    (stable, ordre 2 en temps)
#      θ = 0   → Euler explicite   (conditionnellement stable seulement)
# =============================================================================

import numpy as np
from scipy.sparse.linalg import spsolve


def apply_dirichlet_by_reduction(K, F, dirichlet_dofs, dirichlet_values):
    """
    Applique les conditions de Dirichlet par élimination (réduction du système).

    Principe :
    ----------
    On part de K·U = F. On partitionne les DDLs en deux groupes :
      - DDLs libres    (F) : inconnues à déterminer
      - DDLs Dirichlet (D) : valeurs imposées connues

    Le système par blocs donne :
        [ K_FF  K_FD ] [ U_F ]   [ F_F ]
        [ K_DF  K_DD ] [ U_D ] = [ F_D ]

    En développant la première ligne :
        K_FF · U_F = F_F - K_FD · U_D
                     ^^^^^^^^^^^^^^^^^^^
                     second membre réduit

    Puisque U_D est connu, ce système réduit n'a pour inconnue que U_F.
    On le résout, puis on recolle U_F et U_D dans le vecteur solution global.

    Paramètres
    ----------
    K               : matrice globale (sparse, n * n)
    F               : vecteur second membre global (n,)
    dirichlet_dofs  : indices des DDLs imposés
    dirichlet_values: valeurs imposées sur ces DDLs

    Retour
    ------
    K_FF      : sous-matrice réduite (sparse, n_free * n_free)
    F_red     : second membre réduit (n_free,)
    free_dofs : indices des DDLs libres
    U_full    : vecteur solution pré-rempli avec les valeurs Dirichlet
    """
    dirichlet_dofs   = np.asarray(dirichlet_dofs,   dtype=int)
    dirichlet_values = np.asarray(dirichlet_values, dtype=float)

    n = len(F)

    # Construction du masque des DDLs libres (tout ce qui n'est pas Dirichlet)
    mask = np.ones(n, dtype=bool)
    mask[dirichlet_dofs] = False
    free_dofs = np.nonzero(mask)[0]

    # Extraction des sous-blocs du système
    K_FF = K[free_dofs, :][:, free_dofs]      # lignes et colonnes libres
    K_FD = K[free_dofs, :][:, dirichlet_dofs] # couplage entre libres et imposés

    # Second membre réduit : on déplace la contribution des DDLs imposés
    # du côté droit de l'égalité
    F_red = F[free_dofs] - K_FD.dot(dirichlet_values)

    # Vecteur solution initialisé avec les valeurs Dirichlet déjà en place.
    # Les DDLs libres seront remplis après résolution.
    U_full = np.zeros(n, dtype=float)
    U_full[dirichlet_dofs] = dirichlet_values

    return K_FF, F_red, free_dofs, U_full


def solve_dirichlet(K, F, dirichlet_dofs, dirichlet_values):
    """
    Résout le système K·U = F avec conditions de Dirichlet fortes.

    Utilise apply_dirichlet_by_reduction pour réduire le système,
    résout le système réduit, puis reconstruit le vecteur solution complet.

    Paramètres
    ----------
    K               : matrice globale (sparse)
    F               : vecteur second membre global
    dirichlet_dofs  : indices des DDLs imposés
    dirichlet_values: valeurs imposées

    Retour
    ------
    U_full : vecteur solution complet (DDLs libres + imposés)
    """
    K_red, F_red, free_dofs, U_full = apply_dirichlet_by_reduction(K, F, dirichlet_dofs, dirichlet_values)

    # Résolution du système réduit K_FF · U_F = F_red
    U_full[free_dofs]      = spsolve(K_red.tocsr(), F_red)
    U_full[dirichlet_dofs] = dirichlet_values   # sécurité : on réimpose

    return U_full


def theta_step(M, K, F_n, F_np1, U_n, dt, theta, dirichlet_dofs, dir_vals_np1):
    """
    Effectue un pas de temps du schéma θ pour le problème :
        M · ∂u/∂t + K · u = F(t)

    Discrétisation :
    ----------------
    On évalue le terme spatial à l'instant intermédiaire t^n + θΔt :

        M (u^{n+1} - u^n)/Δt + K (θ u^{n+1} + (1-θ) u^n) = θ F^{n+1} + (1-θ) F^n

    En regroupant u^{n+1} à gauche et u^n à droite :

        A · u^{n+1} = rhs

    avec :
        A   = M + θ Δt K
        rhs = (M - (1-θ) Δt K) · u^n  +  Δt (θ F^{n+1} + (1-θ) F^n)

    Les conditions de Dirichlet sont imposées à l'instant t^{n+1}.

    Paramètres
    ----------
    M             : matrice de masse (sparse)
    K             : matrice de rigidité à l'instant courant (sparse)
    F_n           : vecteur de charge à t^n
    F_np1         : vecteur de charge à t^{n+1}
    U_n           : solution à t^n
    dt            : pas de temps
    theta         : paramètre θ du schéma (0 ≤ θ ≤ 1)
    dirichlet_dofs: indices des DDLs de Dirichlet
    dir_vals_np1  : valeurs Dirichlet imposées à t^{n+1}

    Retour
    ------
    U_np1 : solution à t^{n+1}
    """
    # Construction du système A · u^{n+1} = rhs
    A   = M + theta * dt * K
    B   = M - (1.0 - theta) * dt * K
    rhs = B.dot(U_n) + dt * (theta * F_np1 + (1.0 - theta) * F_n)

    # Réduction et résolution avec les conditions de Dirichlet à t^{n+1}
    A_red, rhs_red, free_dofs, U_np1 = apply_dirichlet_by_reduction(A, rhs, dirichlet_dofs, dir_vals_np1)

    U_np1[free_dofs]      = spsolve(A_red.tocsr(), rhs_red)
    U_np1[dirichlet_dofs] = dir_vals_np1   # sécurité : on réimpose

    return U_np1