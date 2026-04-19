from newton_solver import newton_solver

from stiffness_non_linear import assemble_stiffness_and_rhs

from dirichlet import theta_step

import numpy as np
import argparse
import math
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation, PillowWriter

from gmsh_utils import (
    gmsh_init, open_2d_mesh,
    prepare_quadrature_and_basis, get_jacobians,
    border_dofs_from_tags, gmsh_finalize
)
from mass import assemble_mass
from plot_utils import plot_fe_solution_2d

# =============================================================================
# SECTION 1 — Constantes géographiques
# =============================================================================

CITY_CX, CITY_CY = 23.0, 58.0
CITY_R_HARD = 5.0
CITY_R_SOFT = 12.0

MTN1_CX, MTN1_CY = 35.0, 106.0
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

K_COAST    = 10.0   # capacité très faible au contact de la mer [ind/km²]
COAST_BAND = 0.5    # largeur de la bande côtière défavorable [km]

# ◄◄◄ MODIFICATION 2 — Paramètre de la non-linéarité de κ
#      Loi choisie : κ(u,x) = κ_base(x) / (1 + α·u)
#      Justification : à forte densité, compétition pour l'espace
#      → les individus explorent moins loin → mobilité réduite.
#      α = 0.02 km²/ind : effet perceptible mais pas dominant.
ALPHA_KAPPA = 0.02   # [km²/ind] — intensité de l'effet densité   ◄◄◄ MOD 2


# =============================================================================
# SECTION 3 — Construction du problème
#
# Cette fonction regroupe toute la phase de préparation :
#   - chargement du maillage
#   - construction du mapping Gmsh → DDL compact
#   - calcul des jacobiens et des données de quadrature
#   - identification des conditions aux limites
#   - construction du champ K nodal
#   - assemblage de la matrice de masse
#   - construction de la condition initiale
#
# L'objectif est de séparer complètement :
#   1. la PRÉPARATION du problème
#   2. la SIMULATION (boucle en temps)
#   3. le POST-TRAITEMENT (plots, erreurs, animations)
#
# Ainsi, run_simulation(problem, ...) pourra se contenter de lire les données
# déjà préparées dans le dictionnaire `problem`.
# =============================================================================

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


def build_problem(order=1, msh_filename="invasion_map.msh", gmsh_model_name="invasion_frelon"):
    """
    Construit et retourne toutes les données nécessaires à la simulation.

    Paramètres
    ----------
    order : int
        Ordre polynomial des éléments finis.
    msh_filename : str
        Nom du fichier .msh généré par msh.py.
    gmsh_model_name : str
        Nom du modèle Gmsh à initialiser.

    Retour
    ------
    problem : dict
        Dictionnaire contenant toutes les structures utiles pour :
          - la simulation IMEX
          - la simulation implicite Newton
          - les comparaisons d'erreur
          - les diagnostics
          - le post-traitement
    """

    # ── X.1 Initialisation de Gmsh ─────────────────────────────────────────
    gmsh_init(gmsh_model_name)

    # ── X.2 Chargement du maillage ─────────────────────────────────────────
    elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags = \
        open_2d_mesh(msh_filename=msh_filename, order=order)

    # ── X.3 Construction du mapping tag Gmsh → DDL compact ────────────────
    #
    # Gmsh utilise des tags de nœuds qui ne sont pas forcément contigus.
    # On construit donc :
    #   - tag_to_dof[tag] = indice compact 0..num_dofs-1
    #   - dof_coords[i]   = coordonnées physiques du DDL i
    #
    # Attention :
    #   on n'utilise PAS all_coords[i] directement, car l'ordre interne
    #   des lignes de nodeCoords n'est pas garanti identique à celui des tags.
    unique_dofs_tags = np.unique(elemNodeTags)
    num_dofs = len(unique_dofs_tags)
    max_tag = int(np.max(nodeTags))

    all_coords = nodeCoords.reshape(-1, 3)
    tag_to_dof = np.full(max_tag + 1, -1, dtype=int)
    dof_coords = np.zeros((num_dofs, 3), dtype=float)

    tag_to_node_index = {int(tag): i for i, tag in enumerate(nodeTags)}

    for i, tag in enumerate(unique_dofs_tags):
        tag_int = int(tag)
        tag_to_dof[tag_int] = i
        dof_coords[i] = all_coords[tag_to_node_index[tag_int]]

    # ── X.4 Quadrature et jacobiens ────────────────────────────────────────
    #
    # Ces données géométriques et de quadrature seront réutilisées :
    #   - dans l'assemblage IMEX à chaque pas de temps
    #   - dans le prétraitement du solveur de Newton
    xi, w, N, gN = prepare_quadrature_and_basis(elemType, order)
    jac, det, coords = get_jacobians(elemType, xi)

    # ── X.5 Identification des DDLs de frontière ───────────────────────────
    bnd_names = [name for name, _ in bnds]

    def get_dofs(bnd_name):
        """
        Retourne les DDLs d'une frontière nommée.
        Si la frontière n'existe pas, retourne un tableau vide.
        """
        if bnd_name in bnd_names:
            return border_dofs_from_tags(
                bnds_tags[bnd_names.index(bnd_name)],
                tag_to_dof
            )
        return np.array([], dtype=int)

    outer_dofs = get_dofs("OuterBoundary")

    # ── X.6 Conditions aux limites ─────────────────────────────────────────
    #
    # En Corse :
    #   - la mer est modélisée par Dirichlet u = 0 sur OuterBoundary
    dir_dofs = outer_dofs.astype(int)
    dir_vals = np.zeros(len(dir_dofs), dtype=float)

    # ── X.7 Distance nodale à la côte ──────────────────────────────────────
    #
    # On approxime la côte par les nœuds de la frontière extérieure.
    # Un KDTree permet ensuite de calculer efficacement la distance de chaque
    # nœud intérieur au bord marin.
    coast_xy = dof_coords[outer_dofs, :2]
    coast_tree = cKDTree(coast_xy)
    dist_coast_nodal, _ = coast_tree.query(dof_coords[:, :2])

    # ── X.8 Construction du champ K nodal ──────────────────────────────────
    #
    # K_nodal[i] représente la capacité de charge locale au nœud i.
    # On reconstruit exactement la logique du fichier principal IMEX :
    #   - halo côtier défavorable
    #   - effet urbain hostile
    #   - bonus bocager intérieur
    K_nodal = np.empty(num_dofs, dtype=float)

    for i in range(num_dofs):
        x = dof_coords[i]

        d_coast = dist_coast_nodal[i]
        if d_coast < COAST_BAND:
            t = d_coast / COAST_BAND
            K_base = K_COAST + t * (K_RURAL - K_COAST)
        else:
            K_base = K_RURAL

        dist_city = math.sqrt((x[0] - CITY_CX)**2 + (x[1] - CITY_CY)**2)
        if dist_city < CITY_R_HARD:
            K_city = K_URBAN
        elif dist_city < CITY_R_SOFT:
            t_city = (dist_city - CITY_R_HARD) / (CITY_R_SOFT - CITY_R_HARD)
            K_city = K_URBAN + t_city * (K_RURAL - K_URBAN)
        else:
            K_city = K_RURAL

        K_local = min(K_base, K_city)

        dist_bocage = math.sqrt((x[0] - BOCAGE_CX)**2 + (x[1] - BOCAGE_CY)**2)
        forest_factor = math.exp(-dist_bocage / 25.0)
        K_bonus = (K_FOREST - K_RURAL) * forest_factor

        K_nodal[i] = K_local + K_bonus


    # ── X.10 Masque urbain pour diagnostics ────────────────────────────────
    dist_city_nodal = np.array([
        math.sqrt((dof_coords[i, 0] - CITY_CX)**2 + (dof_coords[i, 1] - CITY_CY)**2)
        for i in range(num_dofs)
    ])
    urban_core_mask = dist_city_nodal < CITY_R_HARD

    # ── X.11 Condition initiale ────────────────────────────────────────────
    #
    # Foyer gaussien centré en (X0, Y0), ensuite tronqué par K_nodal pour
    # garantir u0 ≤ K dès l'instant initial.
    R0 = 5.0
    A0 = 5.0

    U0 = np.array([
        A0 * math.exp(-((dof_coords[i, 0] - X0)**2 + (dof_coords[i, 1] - Y0)**2) / (2 * R0**2))
        for i in range(num_dofs)
    ], dtype=float)

    U0 = np.minimum(U0, K_nodal)
    U0[dir_dofs] = 0.0

    # ── X.12 Assemblage de la matrice de masse ─────────────────────────────
    #
    # La matrice de masse ne dépend ni du temps ni de u.
    # Elle peut donc être assemblée une seule fois ici.
    M_lil = assemble_mass(elemTags, elemNodeTags, det, w, N, tag_to_dof)
    M = M_lil.tocsr()

    # Masse lumpée :
    # utile pour la réaction explicite dans le schéma IMEX
    M_lump = np.array(M.sum(axis=1)).flatten()

    # ── X.13 Prétraitement optionnel pour Newton ───────────────────────────
    #
    # Pour comparer avec le Newton,
    # ce bloc peut être activé.
    #
    # On évite de le rendre obligatoire pour que build_problem puisse aussi
    # servir à une simulation purement IMEX sans dépendance inutile.
    try:
        from newton_solver import preprocess_newton_data

        newton_data = preprocess_newton_data(
            elemTags=elemTags,
            conn=elemNodeTags,
            jac=jac,
            det=det,
            xphys=coords,
            w=w,
            N=N,
            gN=gN,
            tag_to_dof=tag_to_dof,
            K_nodal=K_nodal
        )
     
    except Exception:
        newton_data = None

    # ── X.14 Construction du dictionnaire final ────────────────────────────
    #
    # On retourne toutes les structures utiles pour :
    #   - run_simulation(..., method="imex")
    #   - run_simulation(..., method="implicit")
    #   - comparaison d'erreurs
    #   - diagnostics et post-traitement
    problem = {
    # géométrie / maillage
    "elemType": elemType,
    "nodeTags": nodeTags,
    "nodeCoords": nodeCoords,
    "elemTags": elemTags,
    "elemNodeTags": elemNodeTags,
    "bnds": bnds,
    "bnds_tags": bnds_tags,

    # ddl / coordonnées
    "num_dofs": num_dofs,
    "tag_to_dof": tag_to_dof,
    "dof_coords": dof_coords,

    # quadrature / géométrie EF
    "xi": xi,
    "w": w,
    "N": N,
    "gN": gN,
    "jac": jac,
    "det": det,
    "coords": coords,

    # matrices
    "M": M,
    "M_lump": M_lump,

    # conditions aux limites
    "dir_dofs": dir_dofs,
    "dir_vals": dir_vals,
    "outer_dofs": outer_dofs,

    # champs physiques
    "K_nodal": K_nodal,
    "kappa_fun": kappa_fun,
    "dkappa_du": dkappa_du,
    "R_GROWTH": R_GROWTH,

    # diagnostics
    "urban_core_mask": urban_core_mask,
    "dist_coast_nodal": dist_coast_nodal,
    "dist_city_nodal": dist_city_nodal,

    # condition initiale
    "U0": U0,

    # données Newton
    "newton_data": newton_data,
}

    return problem

def run_simulation(problem, method="implicit", dt=0.5, nsteps=30, save_every=1, theta=1.0):
    M = problem["M"]
    dir_dofs = problem["dir_dofs"]
    dir_vals = problem["dir_vals"]
    urban_core_mask = problem["urban_core_mask"]
    U = problem["U0"].copy()

    saved_times = []
    saved_fields = []
    diagnostics = {
        "u_max": [],
        "u_mean_urban": [],
        "invaded_fraction": [],
    }

    num_dofs = len(U)

    for step in range(nsteps):
        t = step * dt
        U_old = U.copy()

        if method == "newton":
            U = newton_solver(
                U_init=U_old.copy(),
                U_old=U_old,
                M=problem["M"],
                dt=dt,
                newton_data=problem["newton_data"],
                kappa_fun=problem["kappa_fun"],
                dkappa_du=problem["dkappa_du"],
                r_growth=problem["R_GROWTH"],
                dirichlet_dofs=dir_dofs,
                dirichlet_vals=dir_vals
            )

        elif method == "imex":
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

            U_pos = np.maximum(U_old, 0.0)
            K_nodal = problem["K_nodal"]
            M_lump = problem["M_lump"]
            R_GROWTH = problem["R_GROWTH"]

            f_react = R_GROWTH * U_pos * (1.0 - U_pos / K_nodal)
            F_total = F0 + f_react * M_lump

            U = theta_step(
                M, K_mat,
                F_total, F_total,
                U_old,
                dt=dt,
                theta=theta,
                dirichlet_dofs=dir_dofs,
                dir_vals_np1=dir_vals
            )

        else:
            raise ValueError(f"Unknown method: {method}")

        U = np.maximum(U, 0.0)
        U[dir_dofs] = dir_vals

        if step % save_every == 0:
            saved_times.append(t + dt)
            saved_fields.append(U.copy())

        u_core = U[urban_core_mask]
        n_inv = np.sum(U > 1.0)

        diagnostics["u_max"].append(np.max(U))
        diagnostics["u_mean_urban"].append(np.mean(u_core))
        diagnostics["invaded_fraction"].append(n_inv / num_dofs)

        # Affichage de la progression de la simulation tous les 10 steps
        if step % 10 == 0 or step == nsteps - 1:
            progress = 100.0 * (step + 1) / nsteps
            print(f"\r[{method}] Progression : {progress:6.2f}% ({step+1}/{nsteps})", end="")
            print()

    return {
        "times": np.array(saved_times),
        "fields": saved_fields,
        "diagnostics": diagnostics,
        "final_state": U.copy(),
        "method": method,
        "dt": dt,
        "nsteps": nsteps,
    }

# =============================================================================
# SECTION 4 — Visualisation a posteriori
# =============================================================================
CYAN    = '#00E5FF'
GOLD    = '#FFD700'
LORANGE = '#FFAB40'


def add_overlays(ax, t_year):
    theta_arc = np.linspace(0, 2 * math.pi, 300)
    halo = [pe.withStroke(linewidth=2, foreground='black')]

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

def save_results_animation(problem, results, output_file="simulation.mp4", stride=1, fps=10):
    """
    Exporte les snapshots stockés dans `results` sous forme de vidéo MP4.

    Paramètres
    ----------
    problem : dict
        Dictionnaire renvoyé par build_problem().
    results : dict
        Dictionnaire renvoyé par run_simulation().
    output_file : str
        Nom du fichier de sortie.
    stride : int
        Une frame sur `stride` est conservée.
    fps : int
        Nombre d'images par seconde de la vidéo.
        Plus fps est grand, plus la vidéo sera rapide.
    """

    times = results["times"][::stride]
    fields = results["fields"][::stride]

    elemNodeTags = problem["elemNodeTags"]
    nodeTags     = problem["nodeTags"]
    nodeCoords   = problem["nodeCoords"]
    tag_to_dof   = problem["tag_to_dof"]

    c_star = 2.0 * math.sqrt(KAPPA_RURAL * R_GROWTH)

    fig, ax = plt.subplots(figsize=(9, 10))
    fig.patch.set_facecolor('#0d0d1a')
    ax.set_facecolor('#0d0d1a')

    contour_holder = {"obj": None}
    cb_holder = {"obj": None}

    def update(frame_idx):
        t = times[frame_idx]
        U = fields[frame_idx]

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
        contour_holder["obj"] = contour

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

        if cb_holder["obj"] is None:
            cb = fig.colorbar(
                contour, ax=ax,
                label='Densité u [ind/km²]',
                pad=0.02, fraction=0.03
            )
            cb.ax.yaxis.label.set_color('white')
            cb.ax.tick_params(colors='white')
            cb_holder["obj"] = cb

        fig.tight_layout(rect=[0, 0.12, 1, 1])

    anim = FuncAnimation(fig, update, frames=len(times), repeat=False)

    writer = PillowWriter(fps=fps)
    anim.save(output_file, writer=writer)

    plt.close(fig)
    print(f"Animation sauvegardée dans : {output_file}")

# =============================================================================
# SECTION 5 — Programme principal 
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fisher-KPP 2D — Invasion du frelon asiatique (Vespa velutina)"
    )
    parser.add_argument("-order", type=int, default=1,
                        help="Ordre polynomial des éléments")
    parser.add_argument("--theta", type=float, default=1.0,
                        help="Paramètre theta du schéma")
    parser.add_argument("--dt", type=float, default=0.1,
                        help="Pas de temps [années]")
    parser.add_argument("--nsteps", type=int, default=600,
                        help="Nombre de pas de temps")
    parser.add_argument("--method", type=str, default="newton",
                        choices=["imex", "newton"],
                        help="Méthode temporelle")
    parser.add_argument("--save_every", type=int, default=5,
                        help="Sauvegarde un snapshot tous les save_every pas")
    parser.add_argument("--no_visu", action="store_true",
                        help="Ne pas afficher la visualisation finale")
    args = parser.parse_args()

    # ── 5.1 Construction du problème ───────────────────────────────────────
    problem = build_problem(
        order=args.order,
        msh_filename="invasion_map.msh",
        gmsh_model_name="invasion_frelon"
    )

    # ── 5.2 Infos console ──────────────────────────────────────────────────
    c_star = 2.0 * math.sqrt(KAPPA_RURAL * R_GROWTH)

    print(f"\n{'═' * 62}")
    print("  Fisher-KPP — Vespa velutina (Corse)")
    print(f"  κ(u,x) non linéaire : κ_base(x) / (1 + {ALPHA_KAPPA}·u)")
    print(f"  κ rural={KAPPA_RURAL} | κ urbain={KAPPA_URBAN} km²/an")
    print(f"  r={R_GROWTH} an⁻¹")
    print(f"  K côte={K_COAST} → rural={K_RURAL} → bocage={K_FOREST}")
    print(f"  mer : Dirichlet u=0 sur OuterBoundary")
    print(f"  méthode : {args.method}")
    print(f"  dt={args.dt} an | T={args.dt * args.nsteps:.1f} ans | DDLs={problem['num_dofs']}")
    print(f"  c* = {c_star:.2f} km/an")
    print(f"{'═' * 62}\n")

    # ── 5.3 Simulation ─────────────────────────────────────────────────────
    results = run_simulation(
        problem,
        method=args.method,
        dt=args.dt,
        nsteps=args.nsteps,
        save_every=args.save_every,
        theta=args.theta
    )

    # ── 5.4 Résumé console ─────────────────────────────────────────────────
    print("Simulation terminée.")
    print(f"Nombre de snapshots sauvegardés : {len(results['fields'])}")
    print(f"u_max final = {results['diagnostics']['u_max'][-1]:.3f}")
    print(f"fraction envahie finale = {results['diagnostics']['invaded_fraction'][-1]:.3f}")

    # ── 5.5 Visualisation a posteriori ─────────────────────────────────────
    if not args.no_visu:
        save_results_animation(problem, results, output_file="test.gif", stride=1, fps=10)

    gmsh_finalize()
    return problem, results

if __name__ == "__main__":
    main()