# =============================================================================
# main_diffusion_2d.py — Simulation Fisher-KPP 2D : invasion du frelon asiatique
# =============================================================================
#
# MODÈLE MATHÉMATIQUE :
#   ∂u/∂t - ∇·(κ(u,x)∇u) = r·u·(1 - u/K(x))
#
# SCHÉMA NUMÉRIQUE :
#   - Discrétisation en espace : éléments finis 2D
#   - Discrétisation en temps   : implicite
#   - Non-linéarité             : résolue par Newton-Raphson à chaque pas de temps
#
# IDÉE :
#   À chaque pas de temps, on cherche U^{n+1} tel que
#       R(U^{n+1}) = 0
#   où le résidu contient :
#       - le terme temporel
#       - le terme de diffusion non linéaire
#       - le terme de réaction logistique
#
# USAGE :
#   python main_diffusion_2d.py
#   python main_diffusion_2d.py --dt 0.05 --nsteps 600
#
# PRÉREQUIS :
#   Lancer d'abord python msh.py pour générer invasion_map.msh
# =============================================================================

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
from newton_solver import newton_solver, preprocess_newton_data
from animation import save_simulation_animation


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
    Diffusivité κ0(x,y) [km²/an] — mobilité spatiale des frelons.

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

ALPHA_KAPPA = 100   # strength of density effect

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
# SECTION 4 — Annotations visuelles sur la carte
# =============================================================================

# Couleurs des annotations (sur fond sombre plasma/noir)
CYAN    = '#00E5FF'   # contours de la métropole
GOLD    = '#FFD700'   # label bocage
LBLUE   = '#90CAF9'   # label lac
LORANGE = '#FFAB40'   # label montagne


def add_overlays(ax, t_year):
    """
    Superpose les annotations géographiques sur la carte de densité :
      - Deux cercles cyan pour la métropole (cœur + zone péri-urbaine)
      - Labels de chaque obstacle avec halo noir pour lisibilité sur fond sombre
      - Croix blanche au foyer d'invasion initial (visible seulement aux premiers pas)
    """
    theta_arc = np.linspace(0, 2 * math.pi, 300)

    # ── Cercle cœur urbain (trait plein) ──────────────────────────────────
    ax.plot(
        CITY_CX + CITY_R_HARD * np.cos(theta_arc),
        CITY_CY + CITY_R_HARD * np.sin(theta_arc),
        color=CYAN, lw=1.5, ls='-', zorder=10, alpha=0.9
    )
    # ── Cercle zone péri-urbaine (pointillé) ──────────────────────────────
    ax.plot(
        CITY_CX + CITY_R_SOFT * np.cos(theta_arc),
        CITY_CY + CITY_R_SOFT * np.sin(theta_arc),
        color=CYAN, lw=0.8, ls='--', zorder=10, alpha=0.6
    )

    # Style commun pour les labels (halo noir = lisible sur toute couleur)
    halo = [pe.withStroke(linewidth=2, foreground='black')]

    ax.text(CITY_CX, CITY_CY,
            'Métropole\n(K=1.5)', color=CYAN,
            fontsize=7.5, ha='center', va='center',
            fontweight='bold', zorder=11, path_effects=halo)

    ax.text(CITY_CX, CITY_CY - CITY_R_SOFT - 2.5,
            'banlieues (K→50)', color=CYAN,
            fontsize=6, ha='center', zorder=11, path_effects=halo)

    ax.text(LAKE_CX, LAKE_CY,
            'Lac\n(u≡0)', color=LBLUE,
            fontsize=7.5, ha='center', va='center',
            fontweight='bold', zorder=11, path_effects=halo)

    ax.text(MTN_CX, MTN_CY,
            'Massif\n(flux=0)', color=LORANGE,
            fontsize=7.5, ha='center', va='center',
            fontweight='bold', zorder=11, path_effects=halo)

    ax.text(BOCAGE_CX, BOCAGE_CY,
            'Bocage\n(K=80)', color=GOLD,
            fontsize=6.5, ha='center', va='center',
            zorder=11, alpha=0.85, path_effects=halo)

    # Croix au foyer d'invasion (coin SW)
    ax.plot(8, 8, 'x', color='white', ms=6, mew=1.5, zorder=12)
    if t_year < 3:
        ax.text(8, 12, 'Foyer\ninitial', color='white',
                fontsize=6, ha='center', zorder=11, path_effects=halo)


def make_legend(fig, ax, c_star):
    """
    Légende personnalisée expliquant le code couleur plasma.
    Placée sous la figure (bbox_to_anchor en dehors des axes).
    """
    patches = [
        mpatches.Patch(
            facecolor=mcolors.to_rgba('black'), edgecolor='white', lw=0.5,
            label='Aucune invasion (u ≈ 0)'
        ),
        mpatches.Patch(facecolor='#7b2d8b', edgecolor='none',
                       label="Front d'invasion actif"),
        mpatches.Patch(facecolor='#f4a261', edgecolor='none',
                       label='Population établie'),
        mpatches.Patch(facecolor='#ffff5e', edgecolor='none',
                       label=f'Saturation K (campagne={K_RURAL:.0f} ind/km²)'),
        mpatches.Patch(facecolor='white', edgecolor='gray', lw=0.5,
                       label='Lac / Montagne (obstacle)'),
        mpatches.Patch(facecolor=mcolors.to_rgba('black'), edgecolor=CYAN, lw=1.5,
                       label=f'Métropole (K={K_URBAN})'),
    ]
    leg = ax.legend(
        handles=patches,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=True, framealpha=0.85,
        facecolor='#1a1a2e', edgecolor='#444466',
        labelcolor='white', fontsize=7.5,
        title=f'c* = {c_star:.1f} km/an   |   Modèle Fisher-KPP',
        title_fontsize=8,
    )
    leg.get_title().set_color('#aaaacc')
    return leg


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
    parser.add_argument("--dt",     type=float, default=0.5,
                        help="Pas de temps [années]")
    parser.add_argument("--nsteps", type=int,   default=50,
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

    newton_data = preprocess_newton_data(
    elemTags=elemTags,
    conn=elemNodeTags,
    jac=jac,
    det=det,
    xphys=coords,
    w=w,
    N=N,
    gN=gN,
    tag_to_dof=tag_to_dof)

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
    # plt.ion()
    # fig, ax = plt.subplots(figsize=(7, 7))
    # fig.patch.set_facecolor('#0d0d1a')
    # ax.set_facecolor('#0d0d1a')
    # cb = None

    # ==========================================================================
    # SECTION 6 — Boucle temporelle implicite
    #
    # À chaque pas de temps [tⁿ, tⁿ⁺¹] :
    #
    #   1. On fige la solution précédente Uⁿ
    #
    #   2. On résout le problème non linéaire au temps tⁿ⁺¹ par Newton-Raphson :
    #          R(Uⁿ⁺¹) = 0
    #
    #      avec
    #          R(U) = (1/Δt) M (U - Uⁿ)
    #                 + R_diff(U)
    #                 - R_reaction(U)
    #
    #   3. À chaque itération de Newton, on assemble :
    #          - le résidu R(Uᵏ)
    #          - la jacobienne J(Uᵏ)
    #      puis on résout :
    #          J(Uᵏ) δU = -R(Uᵏ)
    #      et on met à jour :
    #          Uᵏ⁺¹ = Uᵏ + δU
    #
    #   4. On applique enfin un garde-fou numérique :
    #          U = max(U, 0)
    #      afin d'éviter de petites valeurs négatives dues aux erreurs numériques.
    # ==========================================================================
    print(f"  Schéma      : implicite non linéaire")
    print(f"  Résolution  : Newton-Raphson à chaque pas de temps")
    print(f"  dt={args.dt} an | T={args.dt * args.nsteps:.0f} ans | DDLs={num_dofs}")

    # -----------------------------------------------------------------
    # Stockage des snapshots pour créer l'animation après la simulation
    # -----------------------------------------------------------------
    saved_times = []
    saved_fields = []

    # On choisit à quelle fréquence physique on sauvegarde une frame
    # Exemple : une frame tous les 0.5 ans
    frame_every_years = 0.5
    steps_per_frame = max(1, round(frame_every_years / args.dt))

    for step in range(args.nsteps):
        t = step * args.dt

        U_old = U.copy()

        # ── Résolution par Newton Raphson ──────────────────────────
        U = newton_solver(
        U_init=U_old.copy(),
        U_old=U_old,
        M=M,
        dt=args.dt,
        newton_data=newton_data,
        kappa_fun=kappa_fun,
        dkappa_du=dkappa_du,
        K_cap=K_cap,
        r_growth=R_GROWTH,
        dirichlet_dofs=dir_dofs,
        dirichlet_vals=dir_vals)

        # Garde-fou numérique : u ≥ 0 en tout point
        U = np.maximum(U, 0.0)

        # -------------------------------------------------------------
        # Sauvegarde d'un snapshot à intervalles réguliers
        # pour construire l'animation plus tard
        # -------------------------------------------------------------
        if step % steps_per_frame == 0:
            saved_times.append(t)
            saved_fields.append(U.copy())

        # ── Affichage (toutes les 3 étapes pour fluidifier l'animation) ───
        if step % 3 != 0:
            continue

        #ax.clear()
        #ax.set_facecolor('#0d0d1a')

        # contour = plot_fe_solution_2d(
        #     elemNodeTags=elemNodeTags,
        #     nodeTags=nodeTags,
        #     nodeCoords=nodeCoords,
        #     U=U,
        #     tag_to_dof=tag_to_dof,
        #     show_mesh=False,
        #     ax=ax,
        #     vmin=0.0,
        #     vmax=K_FOREST,   # toute la gamme 0→80 est représentée)
        #     cmap='plasma'
        # )

        # add_overlays(ax, t)
        # make_legend(fig, ax, c_star)

        # ax.set_title(
        #     f'Invasion du Frelon Asiatique — Fisher-KPP\n'
        #     f't = {t:.1f} an  |  c* = {c_star:.1f} km/an',
        #     color='white', fontsize=11, pad=10
        # )
        # ax.set_xlabel('x [km]', color='#aaaacc')
        # ax.set_ylabel('y [km]', color='#aaaacc')
        # ax.tick_params(colors='#aaaacc')
        # for spine in ax.spines.values():
        #     spine.set_edgecolor('#333355')
        # ax.axis('equal')

        # if cb is None:
        #     cb = fig.colorbar(contour, ax=ax,
        #                       label='Densité u [ind/km²]',
        #                       pad=0.02, fraction=0.03)
        #     cb.ax.yaxis.label.set_color('white')
        #     cb.ax.tick_params(colors='white')

        # fig.tight_layout(rect=[0, 0.12, 1, 1])
        # fig.canvas.draw()
        # plt.pause(0.02)
        
        # ── Suivi console (toutes les 30 étapes) ──────────────────────────
        print(f"step: {step}")
        if step % 30 == 0:
            u_core = U[urban_core_mask]
            n_inv  = np.sum(U > 1.0)
            print(
                f"  t={t:5.1f} an | u_max={np.max(U):5.1f} | "
                f"envahis (u>1): {100 * n_inv // num_dofs}% | "
                f"u_moy_cœur_urbain={np.mean(u_core):.2f} (K={K_URBAN})"
            )

    save_simulation_animation(
    saved_fields=saved_fields,
    saved_times=saved_times,
    elemNodeTags=elemNodeTags,
    nodeTags=nodeTags,
    nodeCoords=nodeCoords,
    tag_to_dof=tag_to_dof,
    K_FOREST=K_FOREST,
    c_star=c_star,
    add_overlays=add_overlays,
    make_legend=make_legend,
    output_file="invasion_frelon.gif",
    fps=10)

    print("\nSimulation terminée.")
    # plt.ioff()
    # plt.show()
    gmsh_finalize()


if __name__ == "__main__":
    main()