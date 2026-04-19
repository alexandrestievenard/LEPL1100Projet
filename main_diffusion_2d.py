# =============================================================================
# main_diffusion_2d.py — Simulation Fisher-KPP 2D : invasion du frelon asiatique
# =============================================================================

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from scipy.spatial import cKDTree

from gmsh_utils import (
    gmsh_init, gmsh_finalize, open_2d_mesh,
    prepare_quadrature_and_basis, get_jacobians,
    border_dofs_from_tags
)
# ◄◄◄ MODIFICATION 1 — Import du nouveau module de rigidité non linéaire
#      (fichier stiffness_non_linear.py fourni par Alexandre — 
from stiffness_non_linear import assemble_stiffness_and_rhs   # ◄◄◄ MOD 1
from mass import assemble_mass
from dirichlet import theta_step
from plot_utils import plot_mesh_2d, plot_fe_solution_2d


# =============================================================================
# SECTION 1 — Constantes géographiques
# =============================================================================
K_URBAN = 1.5
CITIES = [
    ("Ajaccio",       23.0,  58.0, 4.0, 9.0),
    ("Bastia",        70.2,  148.0, 3.5, 8.0),
    ("Porto-Vecchio", 67.0,  25.5, 2.2, 5.0),
    ("Corte",         50.0,  106.25, 1.6, 3.5),
    ("Calvi",         19.5, 132.0, 1.8, 4.0),
]

MTN1_CX, MTN1_CY = 37.0, 114.0
MTN2_CX, MTN2_CY = 44.0, 96.0
MTN3_CX, MTN3_CY = 49.0, 78.0
MTN4_CX, MTN4_CY = 50.6, 68.2
MTN5_CX, MTN5_CY = 51.2, 50.0

BOCAGE_CX, BOCAGE_CY = 68.0, 115.0

X0, Y0 = 52.0, 25.0   # foyer initial


# =============================================================================
# SECTION 2 — Paramètres physiques
# =============================================================================

KAPPA_RURAL = 5.0
KAPPA_URBAN = 0.5

R_GROWTH = 1.0

K_FOREST = 80.0
K_RURAL  = 50.0
K_URBAN  = 1.5

K_COAST    = 10.0   # capacité très faible au contact de la mer [ind/km²]
COAST_BAND = 0.5    # largeur de la bande côtière défavorable [km]

# ◄◄◄ MODIFICATION 2 — Paramètre de la non-linéarité de κ
#      Loi choisie : κ(u,x) = κ_base(x) / (1 + α·u)
#      Justification : à forte densité, compétition pour l'espace
#      → les individus explorent moins loin → mobilité réduite.
#      α = 0.02 km²/ind : effet perceptible mais pas dominant.
ALPHA_KAPPA = 0.02   # [km²/ind] — intensité de l'effet densité   ◄◄◄ MOD 2


# =============================================================================
# SECTION 3 — Champs spatiaux
# =============================================================================

# ◄◄◄ MODIFICATION 3 — Séparation de kappa en deux fonctions
#
#   AVANT : kappa_fun(x)       → une seule fonction, ne dépend que de x
#   APRÈS : kappa_base(x)      → partie spatiale (identique à l'ancienne)
#           kappa_fun(u, x)    → κ non linéaire, évalué avec u^n dans la boucle
#
#   Dans le schéma IMEX, kappa_fun est appelée avec u = U[n] (connu)
#   → le système reste linéaire en U^{n+1}, pas besoin de Newton.

def kappa_base(x):

    kappa_val = KAPPA_RURAL

    for _, cx, cy, r_hard, r_soft in CITIES:
        dist = math.sqrt((x[0] - cx)**2 + (x[1] - cy)**2)

        if dist < r_hard:
            local_kappa = KAPPA_URBAN
        elif dist < r_soft:
            t = (dist - r_hard) / (r_soft - r_hard)
            local_kappa = KAPPA_URBAN + t * (KAPPA_RURAL - KAPPA_URBAN)
        else:
            local_kappa = KAPPA_RURAL

        kappa_val = min(kappa_val, local_kappa)

    return kappa_val


def kappa_fun(u, x):                                           # ◄◄◄ MOD 3
    """
    Diffusivité non linéaire κ(u, x) [km²/an].

    Loi : κ(u,x) = κ_base(x) / (1 + α·u)

    Interprétation :
      - u faible  → κ ≈ κ_base(x)      (mobilité normale)
      - u élevé   → κ diminue          (forte densité = compétition pour l'espace,
                                         les individus se dispersent moins)

    Dans le schéma IMEX, cette fonction est appelée avec u = u^n (solution
    du pas précédent, entièrement connue). Le terme de diffusion reste donc
    linéaire en u^{n+1} → pas besoin de Newton-Raphson.
    """
    u_pos = max(u, 0.0)   # garde-fou : u ne peut pas être négatif
    return kappa_base(x) / (1.0 + ALPHA_KAPPA * u_pos)        # ◄◄◄ MOD 3


# =============================================================================
# SECTION 4 — Annotations visuelles
# =============================================================================

CYAN    = '#00E5FF'
GOLD    = '#FFD700'
LORANGE = '#FFAB40'


def add_overlays(ax, t_year):
    theta_arc = np.linspace(0, 2 * math.pi, 300)
    halo = [pe.withStroke(linewidth=2, foreground='black')]

    # Villes
    for name, cx, cy, r_hard, r_soft in CITIES:
        ax.plot(
            cx + r_hard * np.cos(theta_arc),
            cy + r_hard * np.sin(theta_arc),
            color=CYAN, lw=1.5, ls='-', zorder=10, alpha=0.9
        )
        ax.plot(
            cx + r_soft * np.cos(theta_arc),
            cy + r_soft * np.sin(theta_arc),
            color=CYAN, lw=0.8, ls='--', zorder=10, alpha=0.6
        )

        ax.text(
            cx, cy,
            f'{name}\n(K={K_URBAN})',
            color=CYAN,
            fontsize=6.8,
            ha='center', va='center',
            fontweight='bold',
            zorder=11,
            path_effects=halo
        )

        ax.text(
            cx, cy - r_soft - 2.0,
            'banlieues (K→50)',
            color=CYAN,
            fontsize=5.5,
            ha='center',
            zorder=11,
            path_effects=halo
        )

    # Bocage
    ax.text(BOCAGE_CX, BOCAGE_CY,
            'Bocage\n(K=80)', color=GOLD,
            fontsize=6.5, ha='center', va='center',
            zorder=11, alpha=0.85, path_effects=halo)

    # Massifs
    ax.text(MTN1_CX, MTN1_CY,
            'Massif\n(flux=0)', color=LORANGE,
            fontsize=7.0, ha='center', va='center',
            fontweight='bold', zorder=11, path_effects=halo)

    ax.text(MTN2_CX, MTN2_CY,
            'Massif\n(flux=0)', color=LORANGE,
            fontsize=7.0, ha='center', va='center',
            fontweight='bold', zorder=11, path_effects=halo)

    ax.text(MTN3_CX, MTN3_CY,
            'Massif\n(flux=0)', color=LORANGE,
            fontsize=4.0, ha='center', va='center',
            fontweight='bold', zorder=11, path_effects=halo)
    
    ax.text(MTN4_CX, MTN4_CY,
            'Massif\n(flux=0)', color=LORANGE,
            fontsize=6.0, ha='center', va='center',
            fontweight='bold', zorder=11, path_effects=halo)
    
    ax.text(MTN5_CX, MTN5_CY,
            'Massif\n(flux=0)', color=LORANGE,
            fontsize=4.0, ha='center', va='center',
            fontweight='bold', zorder=11, path_effects=halo)

    # Foyer initial
    ax.plot(X0, Y0, 'x', color='white', ms=6, mew=1.5, zorder=12)
    if t_year < 15:
        ax.text(X0 + 5, Y0 + 5, 'Foyer\ninitial', color='white',
                fontsize=6, ha='center', zorder=11, path_effects=halo)


def make_legend(fig, ax, c_star):
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
        mpatches.Patch(facecolor=mcolors.to_rgba('black'), edgecolor='white', lw=0.5,
                       label='Mer : u = 0'),
        mpatches.Patch(facecolor='white', edgecolor='gray', lw=0.5,
                       label='Massifs : flux = 0'),
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
    parser.add_argument("-order", type=int, default=1)
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--nsteps", type=int, default=600)
    args = parser.parse_args()

    gmsh_init("invasion_frelon")

    # ── 5.1 Chargement du maillage ────────────────────────────────────────
    elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags = \
        open_2d_mesh(msh_filename="invasion_map.msh", order=args.order)

    plot_mesh_2d(elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags)

    # ── 5.2 Mapping tag Gmsh → DDL compact ───────────────────────────────
    unique_dofs_tags = np.unique(elemNodeTags)
    num_dofs = len(unique_dofs_tags)
    max_tag = int(np.max(nodeTags))

    dof_coords = np.zeros((num_dofs, 3))
    all_coords = nodeCoords.reshape(-1, 3)
    tag_to_dof = np.full(max_tag + 1, -1, dtype=int)

    for i, tag in enumerate(unique_dofs_tags):
        tag_to_dof[int(tag)] = i
        dof_coords[i] = all_coords[i]

    # ── 5.3 Quadrature et jacobiens ───────────────────────────────────────
    xi, w, N, gN = prepare_quadrature_and_basis(elemType, args.order)
    jac, det, coords = get_jacobians(elemType, xi)

    # ── 5.4 Identification des DDLs de frontière ──────────────────────────
    bnd_names = [name for name, _ in bnds]

    def get_dofs(bnd_name):
        if bnd_name in bnd_names:
            return border_dofs_from_tags(
                bnds_tags[bnd_names.index(bnd_name)], tag_to_dof
            )
        return np.array([], dtype=int)

    outer_dofs = get_dofs("OuterBoundary")

    # ── 5.5 Distance à la côte ────────────────────────────────────────────
    coast_xy = dof_coords[outer_dofs, :2]
    coast_tree = cKDTree(coast_xy)
    dist_coast_nodal, _ = coast_tree.query(dof_coords[:, :2])

    # ── 5.6 Champ K nodal ─────────────────────────────────────────────────
    K_nodal = np.empty(num_dofs, dtype=float)

    for i in range(num_dofs):
        x = dof_coords[i]

        # Effet côtier
        d_coast = dist_coast_nodal[i]
        if d_coast < COAST_BAND:
            t = d_coast / COAST_BAND
            K_base = K_COAST + t * (K_RURAL - K_COAST)
        else:
            K_base = K_RURAL

        # Effet urbain : on prend la ville la plus pénalisante
        K_city = K_RURAL
        for _, cx, cy, r_hard, r_soft in CITIES:
            dist_city = math.sqrt((x[0] - cx)**2 + (x[1] - cy)**2)

            if dist_city < r_hard:
                local_K = K_URBAN
            elif dist_city < r_soft:
                t_city = (dist_city - r_hard) / (r_soft - r_hard)
                local_K = K_URBAN + t_city * (K_RURAL - K_URBAN)
            else:
                local_K = K_RURAL

            K_city = min(K_city, local_K)

        K_local = min(K_base, K_city)

        # Bonus bocager
        dist_bocage = math.sqrt((x[0] - BOCAGE_CX)**2 + (x[1] - BOCAGE_CY)**2)
        forest_factor = math.exp(-dist_bocage / 25.0)
        K_bonus = (K_FOREST - K_RURAL) * forest_factor

        K_nodal[i] = K_local + K_bonus

    # ── 5.7 Masques urbains pour suivi console ────────────────────────────
    city_core_masks = {}

    for name, cx, cy, r_hard, _ in CITIES:
        dist_city_nodal = np.array([
            math.sqrt((dof_coords[i, 0] - cx)**2 + (dof_coords[i, 1] - cy)**2)
            for i in range(num_dofs)
        ])
        city_core_masks[name] = dist_city_nodal < r_hard

    # ── 5.8 Condition initiale ────────────────────────────────────────────
    R0 = 5.0
    A0 = 5.0

    U = np.array([
        A0 * math.exp(-((dof_coords[i, 0] - X0)**2 + (dof_coords[i, 1] - Y0)**2) / (2 * R0**2))
        for i in range(num_dofs)
    ], dtype=float)
    U = np.minimum(U, K_nodal)

    # ── 5.9 Assemblage de M uniquement (K sera assemblé dans la boucle) ──
    #
    # ◄◄◄ MODIFICATION 4 — Suppression de l'assemblage de K avant la boucle
    #      AVANT : K_lil, F0 = assemble_stiffness_and_rhs(..., kappa_fun, ...)
    #              K_mat = K_lil.tocsr()
    #              → K assemblé une seule fois avec kappa_fun(x)
    #
    #      APRÈS : K n'est plus assemblé ici.
    #              Il sera assemblé à chaque pas de temps dans la boucle
    #              avec kappa_fun(u^n, x), ce qui permet de prendre en compte
    #              la dépendance non linéaire en u.
    #              M reste assemblé ici car il ne dépend pas de u ni du temps.
    M_lil = assemble_mass(elemTags, elemNodeTags, det, w, N, tag_to_dof)  # ◄◄◄ MOD 4
    M = M_lil.tocsr()
    M_lump = np.array(M.sum(axis=1)).flatten()

    # ── 5.10 Conditions aux limites ───────────────────────────────────────
    dir_dofs = outer_dofs.astype(int)
    dir_vals = np.zeros(len(dir_dofs), dtype=float)
    U[dir_dofs] = 0.0

    # ── 5.11 Infos console ────────────────────────────────────────────────
    c_star = 2.0 * math.sqrt(KAPPA_RURAL * R_GROWTH)

    print(f"\n{'═' * 62}")
    print("  Fisher-KPP — Vespa velutina (Corse)")
    print(f"  κ(u,x) non linéaire : κ_base(x) / (1 + {ALPHA_KAPPA}·u)")
    print(f"  κ rural={KAPPA_RURAL} | κ urbain={KAPPA_URBAN} km²/an")
    print(f"  r={R_GROWTH} an⁻¹")
    print(f"  K côte={K_COAST} → rural={K_RURAL} → bocage={K_FOREST}")
    print(f"  mer : Dirichlet u=0 sur OuterBoundary")
    print(f"  Schéma : IMEX avec κ(u^n,x) explicite — système linéaire ✓")
    print(f"  Stabilité : dt·r = {args.dt * R_GROWTH:.2f} < 1 ✓")
    print(f"  c* = {c_star:.2f} km/an")
    print(f"  dt={args.dt} an | T={args.dt * args.nsteps:.1f} ans | DDLs={num_dofs}")
    print(f"{'═' * 62}\n")

    # ── 5.12 Figure interactive ───────────────────────────────────────────
    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 10))
    fig.patch.set_facecolor('#0d0d1a')
    ax.set_facecolor('#0d0d1a')
    cb = None

    # =========================================================================
    # SECTION 6 — Boucle temporelle IMEX avec κ(u^n, x)
    #
    # À chaque pas de temps [tⁿ, tⁿ⁺¹] :
    #
    #   1. ASSEMBLAGE de K avec κ(u^n, x)  ← nouveau par rapport à l'ancienne version
    #      κ est évalué avec u^n (déjà connu) → système reste linéaire en u^{n+1}
    #      C'est l'idée centrale de l'adaptation IMEX du travail de Newton de Alexandre.
    #
    #   2. RÉACTION (explicite) : f_react = r·u^n·(1 - u^n/K)
    #
    #   3. DIFFUSION (implicite) : (M + Δt·K(u^n)) u^{n+1} = M·u^n + Δt·F_total
    #
    #   4. GARDE-FOU : U = max(U, 0)
    # =========================================================================
    """ Voici la solution qui pourrait faire gagner 5 fois plus de temps (voir explication en 6.7 du LateX)
    for step in range(args.nsteps):
        t = step * args.dt

        # ◄◄◄ MODIFICATION 5 — réassemblage tous les 5 pas pour accélérer
        if step % 5 == 0:
            K_lil, F0 = assemble_stiffness_and_rhs(
                elemTags, elemNodeTags, jac, det, coords, w, N, gN,
                U,
                kappa_fun,
                lambda x: 0.0,
                tag_to_dof
            )
            K_mat = K_lil.tocsr()
    """
    for step in range(args.nsteps):
        t = step * args.dt

        # ◄◄◄ MODIFICATION 5 — Assemblage de K à chaque pas avec κ(u^n, x)
        #
        #      U est la solution du pas précédent (u^n).
        #      stiffness_non_linear.assemble_stiffness_and_rhs attend :
        #        - U        : vecteur nodal courant pour interpoler u^n aux pts de Gauss
        #        - kappa_fun: fonction à signature (u, x) au lieu de (x) seulement
        #
        #      Le système (M + Δt·K(u^n))·u^{n+1} = ... reste linéaire
        #      car u^n est un nombre connu à ce stade du calcul.
        K_lil, F0 = assemble_stiffness_and_rhs(    # ◄◄◄ MOD 5
            elemTags, elemNodeTags, jac, det, coords, w, N, gN,
            U,              # u^n — évalué explicitement
            kappa_fun,      # κ(u, x) — signature (u, x) du fichier d'Alexandre
            lambda x: 0.0,
            tag_to_dof
        )
        K_mat = K_lil.tocsr()                       # ◄◄◄ MOD 5

        # Réaction logistique Fisher-KPP (explicite)
        U_pos   = np.maximum(U, 0.0)
        f_react = R_GROWTH * U_pos * (1.0 - U_pos / K_nodal)
        F_total = F0 + f_react * M_lump

        # Diffusion implicite (theta_step identique à l'ancienne version)
        U = theta_step(
            M, K_mat,
            F_total, F_total,
            U,
            dt=args.dt, theta=args.theta,
            dirichlet_dofs=dir_dofs,
            dir_vals_np1=dir_vals
        )

        U = np.maximum(U, 0.0)
        U[dir_dofs] = 0.0
        
        

        ax.clear()
        ax.set_facecolor('#0d0d1a')

        contour = plot_fe_solution_2d(
            elemNodeTags=elemNodeTags,
            nodeTags=nodeTags,
            nodeCoords=nodeCoords,
            U=U,
            tag_to_dof=tag_to_dof,
            show_mesh=True,
            ax=ax,
            vmin=0.0,
            vmax=K_FOREST,
            cmap='plasma'
        )

        add_overlays(ax, t)
        make_legend(fig, ax, c_star)

        ax.set_title(
            f'Invasion du Frelon Asiatique — Fisher-KPP (Corse)\n'
            f't = {t:.1f} an  |  c* = {c_star:.1f} km/an  |  κ(u,x) non linéaire',
            color='white', fontsize=11, pad=10
        )
        ax.set_xlabel('x [km]', color='#aaaacc')
        ax.set_ylabel('y [km]', color='#aaaacc')
        ax.tick_params(colors='#aaaacc')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')
        ax.axis('equal')

        if cb is None:
            cb = fig.colorbar(
                contour, ax=ax,
                label='Densité u [ind/km²]',
                pad=0.02, fraction=0.03
            )
            cb.ax.yaxis.label.set_color('white')
            cb.ax.tick_params(colors='white')

        fig.tight_layout(rect=[0, 0.12, 1, 1])
        fig.canvas.draw()
        plt.pause(0.02)

        if step % 30 == 0:
            n_inv = np.sum(U > 1.0)

            city_stats = " | ".join(
                f"{name}={np.mean(U[mask]):.2f}"
                for name, mask in city_core_masks.items()
            )

            print(
                f"  t={t:5.1f} an | u_max={np.max(U):5.1f} | "
                f"envahis (u>1): {100 * n_inv // num_dofs}% | "
                f"{city_stats} | "
                f"u_moy_côte={np.mean(U[dist_coast_nodal < 5.0]):.2f}"
            )
    print("\nSimulation terminée.")
    plt.ioff()
    plt.show()
    gmsh_finalize()


if __name__ == "__main__":
    main()