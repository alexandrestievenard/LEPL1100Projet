# =============================================================================
# main_diffusion_2d.py — Simulation Fisher-KPP 2D : invasion du frelon asiatique
# =============================================================================
#
# MODÈLE :
#   ∂u/∂t - ∇·(κ(x)∇u) = r·u·(1 - u/K(x))
#
# CHANGEMENTS POUR LA CORSE :
#   - suppression complète du lac
#   - mer modélisée par Dirichlet sur OuterBoundary : u = 0
#   - halo côtier défavorable via K(x) plus faible près de la côte
#   - hotspot bocager intérieur conservé
#   - métropole conservée
#
# PRÉREQUIS :
#   python msh.py
#   python main_diffusion_2d.py
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
from stiffness import assemble_stiffness_and_rhs
from mass import assemble_mass
from dirichlet import theta_step
from plot_utils import plot_mesh_2d, plot_fe_solution_2d


# =============================================================================
# SECTION 1 — Constantes géographiques
# =============================================================================

CITY_CX, CITY_CY = 23.0, 58.0
CITY_R_HARD = 5.0
CITY_R_SOFT = 12.0

MTN1_CX, MTN1_CY = 35.0, 106.0   # pour la visualisation
MTN2_CX, MTN2_CY = 47.0, 72.0
MTN3_CX, MTN3_CY = 54.0, 53.0 

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

# Nouveau : effet côtier
K_COAST = 10.0         # capacité très faible au contact de la mer [ind/km²]
COAST_BAND = 0.5    # largeur de la bande côtière défavorable [km]


# =============================================================================
# SECTION 3 — Champs spatiaux
# =============================================================================

def kappa_fun(x):
    """
    Diffusivité κ(x,y) :
      - faible dans le cœur urbain
      - transition linéaire dans la couronne péri-urbaine
      - rurale ailleurs
    """
    dist = math.sqrt((x[0] - CITY_CX)**2 + (x[1] - CITY_CY)**2)

    if dist < CITY_R_HARD:
        return KAPPA_URBAN

    if dist < CITY_R_SOFT:
        t = (dist - CITY_R_HARD) / (CITY_R_SOFT - CITY_R_HARD)
        return KAPPA_URBAN + t * (KAPPA_RURAL - KAPPA_URBAN)

    return KAPPA_RURAL


# =============================================================================
# SECTION 4 — Annotations visuelles
# =============================================================================

CYAN    = '#00E5FF'
GOLD    = '#FFD700'
LORANGE = '#FFAB40'


def add_overlays(ax, t_year):
    theta_arc = np.linspace(0, 2 * math.pi, 300)
    halo = [pe.withStroke(linewidth=2, foreground='black')]

    # Métropole
    ax.plot(
        CITY_CX + CITY_R_HARD * np.cos(theta_arc),
        CITY_CY + CITY_R_HARD * np.sin(theta_arc),
        color=CYAN, lw=1.5, ls='-', zorder=10, alpha=0.9
    )
    ax.plot(
        CITY_CX + CITY_R_SOFT * np.cos(theta_arc),
        CITY_CY + CITY_R_SOFT * np.sin(theta_arc),
        color=CYAN, lw=0.8, ls='--', zorder=10, alpha=0.6
    )

    ax.text(CITY_CX, CITY_CY,
            'Métropole\n(K=1.5)', color=CYAN,
            fontsize=7.5, ha='center', va='center',
            fontweight='bold', zorder=11, path_effects=halo)

    ax.text(CITY_CX, CITY_CY - CITY_R_SOFT - 2.5,
            'banlieues (K→50)', color=CYAN,
            fontsize=6, ha='center', zorder=11, path_effects=halo)

    # Bocage
    ax.text(BOCAGE_CX, BOCAGE_CY,
            'Bocage\n(K=80)', color=GOLD,
            fontsize=6.5, ha='center', va='center',
            zorder=11, alpha=0.85, path_effects=halo)
    
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
            fontsize=7.0, ha='center', va='center',
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
    parser.add_argument("-order", type=int, default=1,
                        help="Ordre polynomial des éléments (1 ou 2)")
    parser.add_argument("--theta", type=float, default=1.0,
                        help="Schéma θ : 1=Euler implicite, 0.5=Crank-Nicolson")
    parser.add_argument("--dt", type=float, default=0.1,
                        help="Pas de temps [années]. Doit vérifier dt·r < 1")
    parser.add_argument("--nsteps", type=int, default=600,
                        help="Nombre de pas de temps")
    args = parser.parse_args()

    gmsh_init("invasion_frelon")

    # -------------------------------------------------------------------------
    # 5.1 Chargement du maillage
    # -------------------------------------------------------------------------
    elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags = \
        open_2d_mesh(msh_filename="invasion_map.msh", order=args.order)

    plot_mesh_2d(elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags)

    # -------------------------------------------------------------------------
    # 5.2 Mapping tag Gmsh -> DDL compact
    # -------------------------------------------------------------------------
    unique_dofs_tags = np.unique(elemNodeTags)
    num_dofs = len(unique_dofs_tags)
    max_tag = int(np.max(nodeTags))

    dof_coords = np.zeros((num_dofs, 3))
    all_coords = nodeCoords.reshape(-1, 3)
    tag_to_dof = np.full(max_tag + 1, -1, dtype=int)

    for i, tag in enumerate(unique_dofs_tags):
        tag_to_dof[int(tag)] = i
        dof_coords[i] = all_coords[i]

    # -------------------------------------------------------------------------
    # 5.3 Quadrature et jacobiens
    # -------------------------------------------------------------------------
    xi, w, N, gN = prepare_quadrature_and_basis(elemType, args.order)
    jac, det, coords = get_jacobians(elemType, xi)

    # -------------------------------------------------------------------------
    # 5.4 Identification des DDLs de frontière
    # -------------------------------------------------------------------------
    bnd_names = [name for name, _ in bnds]

    def get_dofs(bnd_name):
        if bnd_name in bnd_names:
            return border_dofs_from_tags(
                bnds_tags[bnd_names.index(bnd_name)], tag_to_dof
            )
        return np.array([], dtype=int)

    outer_dofs = get_dofs("OuterBoundary")
    mtn_dofs = get_dofs("Mountains")

    # -------------------------------------------------------------------------
    # 5.5 Distance à la côte
    # -------------------------------------------------------------------------
    coast_xy = dof_coords[outer_dofs, :2]
    coast_tree = cKDTree(coast_xy)
    dist_coast_nodal, _ = coast_tree.query(dof_coords[:, :2])

    # -------------------------------------------------------------------------
    # 5.6 Champ K nodal : côte défavorable + ville + bocage
    # -------------------------------------------------------------------------
    K_nodal = np.empty(num_dofs, dtype=float)

    for i in range(num_dofs):
        x = dof_coords[i]

        # 1) effet côtier : K faible près de la mer, retour progressif vers K_RURAL
        d_coast = dist_coast_nodal[i]
        if d_coast < COAST_BAND:
            t = d_coast / COAST_BAND
            K_base = K_COAST + t * (K_RURAL - K_COAST)
        else:
            K_base = K_RURAL

        # 2) effet urbain : remplace localement K_base si la ville est plus hostile
        dist_city = math.sqrt((x[0] - CITY_CX)**2 + (x[1] - CITY_CY)**2)
        if dist_city < CITY_R_HARD:
            K_city = K_URBAN
        elif dist_city < CITY_R_SOFT:
            t_city = (dist_city - CITY_R_HARD) / (CITY_R_SOFT - CITY_R_HARD)
            K_city = K_URBAN + t_city * (K_RURAL - K_URBAN)
        else:
            K_city = K_RURAL

        # on prend la zone la plus hostile entre côte et ville
        K_local = min(K_base, K_city)

        # 3) hotspot bocager : bonus intérieur
        dist_bocage = math.sqrt((x[0] - BOCAGE_CX)**2 + (x[1] - BOCAGE_CY)**2)
        forest_factor = math.exp(-dist_bocage / 25.0)
        K_bonus = (K_FOREST - K_RURAL) * forest_factor

        K_nodal[i] = K_local + K_bonus

    # -------------------------------------------------------------------------
    # 5.7 Masque urbain pour suivi console
    # -------------------------------------------------------------------------
    dist_city_nodal = np.array([
        math.sqrt((dof_coords[i, 0] - CITY_CX)**2 + (dof_coords[i, 1] - CITY_CY)**2)
        for i in range(num_dofs)
    ])
    urban_core_mask = dist_city_nodal < CITY_R_HARD

    # -------------------------------------------------------------------------
    # 5.8 Condition initiale
    # -------------------------------------------------------------------------
    R0 = 5.0
    A0 = 5.0

    U = np.array([
        A0 * math.exp(-((dof_coords[i, 0] - X0)**2 + (dof_coords[i, 1] - Y0)**2) / (2 * R0**2))
        for i in range(num_dofs)
    ], dtype=float)

    U = np.minimum(U, K_nodal)

    # -------------------------------------------------------------------------
    # 5.9 Assemblage des matrices
    # -------------------------------------------------------------------------
    K_lil, F0 = assemble_stiffness_and_rhs(
        elemTags, elemNodeTags, jac, det, coords, w, N, gN,
        kappa_fun,
        lambda x: 0.0,
        tag_to_dof
    )
    M_lil = assemble_mass(elemTags, elemNodeTags, det, w, N, tag_to_dof)

    K_mat = K_lil.tocsr()
    M = M_lil.tocsr()
    M_lump = np.array(M.sum(axis=1)).flatten()

    # -------------------------------------------------------------------------
    # 5.10 Conditions aux limites
    # -------------------------------------------------------------------------
    # Mer = mort immédiate
    dir_dofs = outer_dofs.astype(int)
    dir_vals = np.zeros(len(dir_dofs), dtype=float)

    # appliquer dès t=0
    U[dir_dofs] = 0.0

    # -------------------------------------------------------------------------
    # 5.11 Infos console
    # -------------------------------------------------------------------------
    c_star = 2.0 * math.sqrt(KAPPA_RURAL * R_GROWTH)

    print(f"\n{'═' * 62}")
    print("  Fisher-KPP — Vespa velutina")
    print(f"  κ rural={KAPPA_RURAL} | κ urbain={KAPPA_URBAN} km²/an")
    print(f"  r={R_GROWTH} an⁻¹")
    print(f"  K côte={K_COAST} → rural={K_RURAL} → bocage={K_FOREST}")
    print(f"  bande côtière = {COAST_BAND} km")
    print(f"  mer : Dirichlet u=0 sur OuterBoundary")
    print(f"  c* = {c_star:.2f} km/an")
    print(f"  dt={args.dt} an | T={args.dt * args.nsteps:.1f} ans | DDLs={num_dofs}")
    print(f"{'═' * 62}\n")

    # -------------------------------------------------------------------------
    # 5.12 Figure interactive
    # -------------------------------------------------------------------------
    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 10))
    fig.patch.set_facecolor('#0d0d1a')
    ax.set_facecolor('#0d0d1a')
    cb = None

    # -------------------------------------------------------------------------
    # 6 — Boucle temporelle IMEX
    # -------------------------------------------------------------------------
    for step in range(args.nsteps):
        t = step * args.dt

        # Réaction explicite
        U_pos = np.maximum(U, 0.0)
        f_react = R_GROWTH * U_pos * (1.0 - U_pos / K_nodal)
        F_total = F0 + f_react * M_lump

        # Diffusion implicite
        U = theta_step(
            M, K_mat,
            F_total, F_total,
            U,
            dt=args.dt, theta=args.theta,
            dirichlet_dofs=dir_dofs,
            dir_vals_np1=dir_vals
        )

        # garde-fous
        U = np.maximum(U, 0.0)
        U[dir_dofs] = 0.0

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
            vmax=K_FOREST,
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
            u_core = U[urban_core_mask]
            n_inv = np.sum(U > 1.0)
            print(
                f"  t={t:5.1f} an | u_max={np.max(U):5.1f} | "
                f"envahis (u>1): {100 * n_inv // num_dofs}% | "
                f"u_moy_cœur_urbain={np.mean(u_core):.2f} | "
                f"u_moy_côte={np.mean(U[dist_coast_nodal < 5.0]):.2f}"
            )

    print("\nSimulation terminée.")
    plt.ioff()
    plt.show()
    gmsh_finalize()


if __name__ == "__main__":
    main()