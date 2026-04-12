# =============================================================================
# main_diffusion_2d.py — Simulation Fisher-KPP 2D : invasion du frelon asiatique
# =============================================================================
#
# MODÈLE MATHÉMATIQUE :
#   ∂u/∂t - ∇·(κ(x)∇u) = r·u·(1 - u/K(x))
#
# SCHÉMA NUMÉRIQUE : IMEX (IMplicit-EXplicit)
#   - Diffusion : implicite → stabilité inconditionnelle (pas de contrainte sur Δt)
#   - Réaction  : explicite → simple évaluation algébrique, stable si Δt·r < 1
#
# USAGE :
#   python main_diffusion_2d.py                          (paramètres par défaut)
#   python main_diffusion_2d.py --dt 0.05 --nsteps 600  (plus fin en temps)
#   python main_diffusion_2d.py --theta 0.5             (Crank-Nicolson)
#
# PRÉREQUIS :
#   Lancer d'abord python msh.py pour générer invasion_map.msh
# =============================================================================

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe

from gmsh_utils import (
    gmsh_init, gmsh_finalize, open_2d_mesh,
    prepare_quadrature_and_basis, get_jacobians,
    border_dofs_from_tags
)
from stiffness_non_linear import assemble_stiffness_and_rhs
from mass import assemble_mass
from dirichlet import theta_step
from plot_utils import plot_mesh_2d, plot_fe_solution_2d


# =============================================================================
# SECTION 1 — Constantes géographiques
# (doivent être cohérentes avec les positions définies dans msh.py)
# =============================================================================

CITY_CX,   CITY_CY   = 65.0, 30.0   # centre de la Métropole Centrale [km]
CITY_R_HARD          =  8.0          # rayon du cœur urbain dense [km]
CITY_R_SOFT          = 15.0          # rayon de la zone péri-urbaine (banlieues) [km]

LAKE_CX,   LAKE_CY   = 22.0, 68.0   # centre du Lac de la Garenne [km]
MTN_CX,    MTN_CY    = 76.0, 70.0   # centroïde du Massif des Crêtes [km]
BOCAGE_CX, BOCAGE_CY = 30.0, 75.0   # centre du hotspot bocager [km]


# =============================================================================
# SECTION 2 — Paramètres physiques du modèle (Vespa velutina)
# =============================================================================

# Diffusivité κ [km²/an] — mobilité spatiale des frelons
KAPPA_RURAL = 5.0    # En campagne/bocage : déplacements libres
KAPPA_URBAN = 0.5    # En cœur urbain    : béton, lumières, obstacles (×10 plus lent)
                     # → vitesse théorique locale : c*(ville) = 2√(0.5×1) ≈ 1.41 km/an
                     #   contre c*(campagne) = 2√(5×1) ≈ 4.47 km/an

# Taux de croissance intrinsèque r [an⁻¹]
R_GROWTH = 1.0       # Une génération par an, taux de survie ~63%

# Capacité de charge K [ind/km²] — densité maximale supportable par l'environnement
K_FOREST = 80.0      # Bocage/forêt : abondance de ressources, zones de nidification
K_RURAL  = 50.0      # Campagne standard
K_URBAN  =  1.5      # Cœur urbain  : béton, pollution, pesticides (milieu très hostile)


# =============================================================================
# SECTION 3 — Champs spatiaux hétérogènes κ(u,x) et K(x)
# =============================================================================

def kappa_base(x):
    """
    Diffusivité κ(x,y) [km²/an] — mobilité spatiale des frelons.

    Structure :
      dist < R_HARD : κ_urbain = 0.5  (cœur urbain, déplacements très réduits)
      R_HARD ≤ dist < R_SOFT : transition linéaire 0.5 → 5.0
      dist ≥ R_SOFT : κ_rural = 5.0  (campagne)

    La transition linéaire entre les deux paliers évite une discontinuité
    brutale dans κ qui provoquerait des instabilités numériques à l'assemblage
    de la matrice de rigidité K (intégration de κ·∇N·∇N).
    """
    dist = math.sqrt((x[0] - CITY_CX)**2 + (x[1] - CITY_CY)**2)

    if dist < CITY_R_HARD:
        return KAPPA_URBAN

    if dist < CITY_R_SOFT:
        t = (dist - CITY_R_HARD) / (CITY_R_SOFT - CITY_R_HARD)
        return KAPPA_URBAN + t * (KAPPA_RURAL - KAPPA_URBAN)

    return KAPPA_RURAL

ALPHA_KAPPA = 0.02   # strength of density effect

def kappa_fun(u, x):
    """
    Nonlinear diffusivity kappa(u, x).

    Simple test law:
        kappa(u, x) = kappa_base(x) / (1 + alpha * u)

    Meaning:
    - low density  -> diffusion close to the original one
    - high density -> diffusion slows down
    """
    u_pos = max(u, 0.0)   # small safeguard
    return kappa_base(x) / (1.0 + ALPHA_KAPPA * u_pos)

def dkappa_du(u, x):
    """
    Derivative of kappa(u, x) with respect to u.
    """
    u_pos = max(u, 0.0)
    kb = kappa_base(x)
    return -ALPHA_KAPPA * kb / (1.0 + ALPHA_KAPPA * u_pos)**2


def K_cap(x):
    """
    Capacité de charge locale K(x,y) [ind/km²].

    Structure (du centre de la ville vers l'extérieur) :
      dist < R_HARD  : K_URBAN = 1.5     (cœur urbain, très hostile)
      R_HARD ≤ dist < R_SOFT : transition linéaire 1.5 → 50.0 (banlieues)
      dist ≥ R_SOFT  : K_RURAL + (K_FOREST - K_RURAL)·exp(-d_bocage/25)
                        (campagne avec hotspot bocager allant jusqu'à 80)

    La transition en banlieue va directement de K_URBAN=1.5 à K_RURAL=50
    (sans palier intermédiaire) pour rester cohérent avec les paramètres
    physiques du modèle.

    Le hotspot bocager (décroissance exponentielle autour de (30,75)) crée
    un gradient spatial lisse qui valide les critères d'hétérogénéité du projet
    et évite toute discontinuité numérique.
    """
    dist_city   = math.sqrt((x[0] - CITY_CX)**2   + (x[1] - CITY_CY)**2)
    dist_bocage = math.sqrt((x[0] - BOCAGE_CX)**2 + (x[1] - BOCAGE_CY)**2)

    # ── Cœur urbain ───────────────────────────────────────────────────────
    if dist_city < CITY_R_HARD:
        return K_URBAN   # 1.5 ind/km²

    # ── Zone péri-urbaine : transition linéaire 1.5 → 50.0 ───────────────
    if dist_city < CITY_R_SOFT:
        t = (dist_city - CITY_R_HARD) / (CITY_R_SOFT - CITY_R_HARD)   # t ∈ [0, 1]
        return K_URBAN + t * (K_RURAL - K_URBAN)   # de 1.5 à 50 ind/km²

    # ── Campagne avec hotspot bocager ─────────────────────────────────────
    # K = 50 + (80 - 50)·exp(-d_bocage/25)
    # → K = 80 au centre du bocage, K → 50 au loin
    forest_factor = math.exp(-dist_bocage / 25.0)
    return K_RURAL + (K_FOREST - K_RURAL) * forest_factor




# =============================================================================
# SECTION 5 — Programme principal
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fisher-KPP 2D — Invasion du frelon asiatique (Vespa velutina)"
    )
    parser.add_argument("-order",   type=int,   default=1,
                        help="Ordre polynomial des éléments (1 ou 2)")
    parser.add_argument("--theta",  type=float, default=1.0,
                        help="Schéma θ : 1=Euler implicite, 0.5=Crank-Nicolson")
    parser.add_argument("--dt",     type=float, default=0.1,
                        help="Pas de temps [années]. Doit vérifier dt·r < 1 (ici dt<1)")
    parser.add_argument("--nsteps", type=int,   default=450,
                        help="Nombre de pas de temps. T_total = dt × nsteps")
    args = parser.parse_args()

    gmsh_init("invasion_frelon")

    # ── 5.1 Chargement du maillage ─────────────────────────────────────────
    elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags = \
        open_2d_mesh(msh_filename="invasion_map.msh", order=args.order)

    plot_mesh_2d(elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags)

    # ── 5.2 Construction du mapping tag Gmsh → indice DDL compact ─────────
    # Gmsh numérote ses nœuds de façon non-contiguë (les tags peuvent avoir des
    # trous). On construit une numérotation compacte 0..num_dofs-1 pour nos matrices.
    unique_dofs_tags = np.unique(elemNodeTags)
    num_dofs = len(unique_dofs_tags)
    max_tag  = int(np.max(nodeTags))

    dof_coords = np.zeros((num_dofs, 3))
    all_coords = nodeCoords.reshape(-1, 3)
    tag_to_dof = np.full(max_tag + 1, -1, dtype=int)   # -1 = tag non utilisé

    for i, tag in enumerate(unique_dofs_tags):
        tag_to_dof[int(tag)] = i
        dof_coords[i] = all_coords[i]

    # ── 5.3 Quadrature et jacobiens ────────────────────────────────────────
    xi, w, N, gN     = prepare_quadrature_and_basis(elemType, args.order)
    jac, det, coords = get_jacobians(elemType, xi)

    # ── 5.4 Identification des DDLs de chaque frontière ───────────────────
    bnd_names = [name for name, _ in bnds]

    def get_dofs(bnd_name):
        """Retourne les DDLs d'une frontière par son nom (tableau vide si absente)."""
        if bnd_name in bnd_names:
            return border_dofs_from_tags(
                bnds_tags[bnd_names.index(bnd_name)], tag_to_dof
            )
        return np.array([], dtype=int)

    lake_dofs = get_dofs("Lake")
    # Mountains et OuterBoundary → Neumann=0 naturel, aucun DDL à extraire

    # ── 5.5 Précalcul des champs spatiaux aux nœuds ────────────────────────
    # Ces champs ne dépendent pas du temps → calculés une seule fois
    K_nodal = np.array([K_cap(dof_coords[i])   for i in range(num_dofs)])

    # Masque pour le suivi de la densité dans le cœur urbain (console)
    dist_city_nodal = np.array([
        math.sqrt((dof_coords[i, 0] - CITY_CX)**2 + (dof_coords[i, 1] - CITY_CY)**2)
        for i in range(num_dofs)
    ])
    urban_core_mask = dist_city_nodal < CITY_R_HARD

    # ── 5.6 Condition initiale : foyer gaussien au coin SW ─────────────────
    # Gaussienne 2D centrée en (8,8) km, légèrement à l'intérieur du domaine
    # pour ne pas être au contact du bord (qui a Neumann=0 mais est irrégulier).
    # La fonction min(·, K(x)) garantit u0 ≤ K partout dès t=0.
    X0, Y0 = 8.0, 8.0   # position du foyer [km]
    R0     = 5.0          # rayon caractéristique [km]
    A0     = 5.0          # amplitude maximale [ind/km²]

    def u0(x):
        dist2 = (x[0] - X0)**2 + (x[1] - Y0)**2
        return min(A0 * math.exp(-dist2 / (2 * R0**2)), K_cap(x))

    # ── 5.7 Assemblage des matrices (time-independent) ─────────────────────
    # on assemble M
    # une seule fois avant la boucle temporelle (gain de temps majeur).
    M_lil = assemble_mass(elemTags, elemNodeTags, det, w, N, tag_to_dof)
    M      = M_lil.tocsr()

    # Masse lumpée : M_lump[i] = somme de la ligne i de M
    # Utilisée pour évaluer la réaction nœud par nœud sans multiplication matricielle
    M_lump = np.array(M.sum(axis=1)).flatten()

    # ── 5.8 Conditions aux limites ─────────────────────────────────────────
    # Lac → Dirichlet u=0 (densité imposée nulle, barrière létale)
    # OuterBoundary + Mountains → Neumann flux=0 (condition naturelle, rien à faire)
    dir_dofs = lake_dofs.astype(int)
    dir_vals = np.zeros(len(dir_dofs), dtype=float)

    # ── 5.9 Initialisation du vecteur solution ────────────────────────────
    U = np.array([u0(dof_coords[i]) for i in range(num_dofs)], dtype=float)
    U[dir_dofs] = dir_vals   # appliquer Dirichlet dès t=0

    # Vitesse théorique du front (pour campagne homogène)
    c_star = 2.0 * math.sqrt(KAPPA_RURAL * R_GROWTH)

    print(f"\n{'═' * 62}")
    print(f"  Fisher-KPP — Vespa velutina")
    print(f"  κ rural={KAPPA_RURAL} | κ urbain={KAPPA_URBAN} km²/an")
    print(f"  r={R_GROWTH} an⁻¹  |  K urbain={K_URBAN} → rural={K_RURAL} → bocage={K_FOREST}")
    print(f"  c* = {c_star:.2f} km/an  →  invasion complète en ~{100/c_star:.0f} ans")
    print(f"  Schéma IMEX : diffusion implicite (θ={args.theta}), réaction explicite")
    print(f"  Stabilité   : dt·r = {args.dt * R_GROWTH:.2f} < 1 ✓")
    print(f"  dt={args.dt} an | T={args.dt * args.nsteps:.0f} ans | DDLs={num_dofs}")
    print(f"{'═' * 62}\n")

    # ── 5.10 Figure interactive (fond sombre cohérent avec colormap plasma) ─
    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 10))
    fig.patch.set_facecolor('#0d0d1a')
    ax.set_facecolor('#0d0d1a')
    cb = None

    # ==========================================================================
    # SECTION 6 — Boucle temporelle IMEX
    #
    # À chaque pas de temps [tⁿ, tⁿ⁺¹] :
    #
    #   1. RÉACTION (explicite) :
    #      f_react[i] = r · U[i] · (1 - U[i] / K[i])   pour chaque nœud i
    #      F_total = F0 + f_react · M_lump
    #      (F0 ≈ 0 car pas de source volumique imposée)
    #
    #   2. DIFFUSION (implicite via theta_step) :
    #      (M + Δt·K) · Uⁿ⁺¹ = M·Uⁿ + Δt·F_total
    #      avec Dirichlet sur le lac (u=0 maintenu à chaque pas)
    #
    #   3. GARDE-FOU : U = max(U, 0) — la densité ne peut jamais être négative
    # ==========================================================================
    for step in range(args.nsteps):
        t = step * args.dt

        # Assemblage de la matrice de raideur
        K_lil, F0 = assemble_stiffness_and_rhs(
            elemTags, elemNodeTags, jac, det, coords, w, N, gN,
            U,                 # current solution used to evaluate kappa(u,x)
            kappa_fun,
            lambda x: 0.0,
            tag_to_dof
        )
        K_mat = K_lil.tocsr()

        # ── Terme de réaction logistique Fisher-KPP (explicite) ───────────
        U_pos   = np.maximum(U, 0.0)
        f_react = R_GROWTH * U_pos * (1.0 - U_pos / K_nodal)
        F_total = F0 + f_react * M_lump

        # ── Pas de temps θ (diffusion implicite) ──────────────────────────
        U = theta_step(
            M, K_mat,
            F_total, F_total,   # Fn = Fnp1 car F ne dépend pas du temps
            U,
            dt=args.dt, theta=args.theta,
            dirichlet_dofs=dir_dofs,
            dir_vals_np1=dir_vals
        )

        # Garde-fou numérique : u ≥ 0 en tout point
        U = np.maximum(U, 0.0)

        # ── Affichage (toutes les 3 étapes pour fluidifier l'animation) ───
        if step % 3 != 0:
            continue

        ax.clear()
        ax.set_facecolor('#0d0d1a')

        contour = plot_fe_solution_2d(
            elemNodeTags=elemNodeTags,
            nodeTags=nodeTags,
            nodeCoords=nodeCoords,
            U=U,
            tag_to_dof=tag_to_dof,
            show_mesh=False,
            ax=ax,
            vmin=0.0,
            vmax=K_FOREST,   # toute la gamme 0→80 est représentée)
            cmap='plasma'
        )

        add_overlays(ax, t)
        make_legend(fig, ax, c_star)

        ax.set_title(
            f'Invasion du Frelon Asiatique — Fisher-KPP\n'
            f't = {t:.1f} an  |  c* = {c_star:.1f} km/an',
            color='white', fontsize=11, pad=10
        )
        ax.set_xlabel('x [km]', color='#aaaacc')
        ax.set_ylabel('y [km]', color='#aaaacc')
        ax.tick_params(colors='#aaaacc')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')
        ax.axis('equal')

        if cb is None:
            cb = fig.colorbar(contour, ax=ax,
                              label='Densité u [ind/km²]',
                              pad=0.02, fraction=0.03)
            cb.ax.yaxis.label.set_color('white')
            cb.ax.tick_params(colors='white')

        fig.tight_layout(rect=[0, 0.12, 1, 1])
        fig.canvas.draw()
        plt.pause(0.02)

        # ── Suivi console (toutes les 30 étapes) ──────────────────────────
        if step % 30 == 0:
            u_core = U[urban_core_mask]
            n_inv  = np.sum(U > 1.0)
            print(
                f"  t={t:5.1f} an | u_max={np.max(U):5.1f} | "
                f"envahis (u>1): {100 * n_inv // num_dofs}% | "
                f"u_moy_cœur_urbain={np.mean(u_core):.2f} (K={K_URBAN})"
            )

    print("\nSimulation terminée.")
    plt.ioff()
    plt.show()
    gmsh_finalize()


if __name__ == "__main__":
    main()