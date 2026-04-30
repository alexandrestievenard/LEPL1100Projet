# =============================================================================
# runsimulation.py — Point d'entrée unique de la simulation Fisher-KPP
# =============================================================================
#
# USAGE :
#   python msh.py                          (une seule fois, génère invasion_map.msh)
#   python runsimulation.py [options]
#
# OPTIONS :
#   --method     imex | newton   méthode temporelle (défaut : imex)
#   --dt         pas de temps en années (défaut : 0.1)
#   --nsteps     nombre de pas de temps (défaut : 600 → 60 ans)
#   --theta      paramètre θ du schéma (défaut : 1.0 = Euler implicite)
#   --save_every sauvegarde 1 snapshot tous les N pas (défaut : 5)
#   --live       affichage en temps réel pendant le calcul
#   --no_visu    ne génère pas de GIF à la fin
#
# ARCHITECTURE :
#   build_problem()          → charge le maillage, assemble les matrices,
#                              construit K_nodal et la condition initiale
#   run_simulation()         → boucle temporelle, appelle imex_step ou newton_solver
#   save_results_animation() → relit les snapshots et génère le GIF a posteriori
#
# FICHIERS REQUIS :
#   invasion_map.msh  (généré par msh.py)
#   gmsh_utils.py, mass.py, imex_solver.py, newton_solver.py, plot_utils.py
# =============================================================================

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial import cKDTree

from gmsh_utils import (
    gmsh_init, gmsh_finalize, open_2d_mesh,
    prepare_quadrature_and_basis, get_jacobians,
    border_dofs_from_tags
)
from mass import assemble_mass
from imex_solver import imex_step
from newton_solver import newton_solver
from plot_utils import plot_fe_solution_2d


# =============================================================================
# SECTION 1 — Constantes géographiques
# =============================================================================

# Chaque ville est décrite par (nom, cx, cy, r_hard, r_soft) en km.
#   cx, cy  : centre de la ville dans le repère du maillage
#   r_hard  : rayon du cœur urbain — κ = κ_urbain, K = K_urbain
#   r_soft  : rayon de la banlieue — transition linéaire vers les valeurs rurales
CITIES = [
    ("Ajaccio",       23.0,  58.0,   4.0,  9.0),
    ("Bastia",        70.2,  148.0,  3.5,  8.0),
    ("Porto-Vecchio", 67.0,  25.5,   2.2,  5.0),
    ("Corte",         50.0,  106.25, 1.6,  3.5),
    ("Calvi",         19.5,  132.0,  1.8,  4.0),
]

# Centres approximatifs des massifs montagneux (uniquement pour les annotations visuelles)
MTN1_CX, MTN1_CY = 37.0, 114.0   # Monte Cinto
MTN2_CX, MTN2_CY = 44.0, 96.0    # Monte Rotondo
MTN3_CX, MTN3_CY = 49.0, 78.0    # Monte d'Oro
MTN4_CX, MTN4_CY = 50.6, 68.2    # Monte Renoso
MTN5_CX, MTN5_CY = 51.2, 50.0    # Monte Incudine

BOCAGE_CX, BOCAGE_CY = 68.0, 115.0   # centre du hotspot bocager (nord-est)

X0, Y0 = 52.0, 25.0   # point d'introduction de l'invasion (sud de l'île)


# =============================================================================
# SECTION 2 — Paramètres physiques
# =============================================================================

KAPPA_RURAL = 5.0    # diffusivité en campagne [km²/an]
                     # → vitesse théorique du front : c* = 2√(κ·r) ≈ 4.47 km/an
KAPPA_URBAN = 0.5    # diffusivité en ville [km²/an] — béton, éclairage : dispersion réduite

R_GROWTH = 1.0       # taux de croissance intrinsèque r [an⁻¹]
                     # → en l'absence de compétition, la population ×e chaque année

K_FOREST = 80.0      # capacité de charge maximale (bocage) [ind/km²]
K_RURAL  = 50.0      # capacité rurale de base [ind/km²]
K_URBAN  = 1.5       # capacité urbaine [ind/km²] — milieu très hostile

K_COAST    = 10.0    # capacité au contact de la mer [ind/km²]
COAST_BAND = 0.5     # largeur de la bande côtière défavorable [km]
                     # (les frelons côtiers sont exposés aux embruns, vent, etc.)

ALPHA_KAPPA = 0.02   # intensité de la dépendance de κ à la densité [km²/ind]
                     # loi : κ(u,x) = κ_base(x) / (1 + α·u)
                     # → à forte densité, compétition pour l'espace → mobilité réduite


# =============================================================================
# SECTION 3 — Champs spatiaux hétérogènes κ(u,x) et K(x)
# =============================================================================

def kappa_base(x):
    """
    Diffusivité spatiale de base κ₀(x) [km²/an], sans dépendance à u.

    Pour chaque ville, on calcule la diffusivité locale et on retient
    le minimum (ville la plus pénalisante). Cela permet de modéliser
    5 zones urbaines simultanément sans conditions spéciales.

    Profil pour chaque ville :
      dist < r_hard            → κ_urbain = 0.5   (cœur urbain dense)
      r_hard ≤ dist < r_soft   → interpolation linéaire 0.5 → 5.0 (banlieue)
      dist ≥ r_soft            → κ_rural = 5.0    (campagne)
    """
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


def kappa_fun(u, x):
    """
    Diffusivité non linéaire complète κ(u, x) [km²/an].

    κ(u, x) = κ_base(x) / (1 + α·u)

    À faible densité (u ≈ 0) : κ ≈ κ_base(x) → mobilité normale.
    À forte densité (u → K)  : κ diminue → les individus se déplacent moins
    car chaque territoire est déjà occupé (compétition intra-spécifique).

    Dans le schéma IMEX, appelée avec u = u^n (connu) pour garder le
    système linéaire en u^{n+1}.
    Dans Newton, appelée avec u = U^{n+1,(k)} à chaque itération.
    """
    u_pos = max(u, 0.0)
    return kappa_base(x) / (1.0 + ALPHA_KAPPA * u_pos)


def dkappa_du(u, x):
    """

    Nécessaire uniquement pour Newton, qui l'utilise pour construire
    la jacobienne du terme de diffusion (second terme de J2).
    Non utilisée par l'IMEX.
    """
    u_pos = max(u, 0.0)
    return -ALPHA_KAPPA * kappa_base(x) / (1.0 + ALPHA_KAPPA * u_pos)**2


# =============================================================================
# SECTION 4 — Construction du problème
# =============================================================================

def build_problem(order=1, msh_filename="invasion_map.msh",
                  gmsh_model_name="invasion_frelon"):
    """
    Charge le maillage et prépare toutes les structures nécessaires
    à la simulation. Cette fonction n'est appelée qu'une seule fois.

    Elle retourne un dictionnaire `problem` qui regroupe :
      - les données brutes du maillage (connectivité, coordonnées, ...)
      - les données de quadrature (points de Gauss, jacobiens, ...)
      - les matrices M et M_lump (assemblées une seule fois)
      - les conditions aux limites (dir_dofs, dir_vals)
      - les champs physiques (K_nodal, kappa_fun, dkappa_du, R_GROWTH)
      - la condition initiale U0
      - les masques de diagnostic par ville
      - les données pré-calculées pour Newton (newton_data)
    """

    gmsh_init(gmsh_model_name)
    elemType, nodeTags, nodeCoords, elemTags, elemNodeTags, bnds, bnds_tags = open_2d_mesh(msh_filename=msh_filename, order=order)

    # Construction du mapping tag Gmsh → indice DDL compac
    #
    # Problème : Gmsh numérote les nœuds avec des "tags" qui peuvent
    # commencer à 1 et avoir des trous (1, 2, 5, 7, ...).
    # Nos matrices numpy ont besoin d'indices contigus (0, 1, 2, 3, ...).
    #
    # Solution : on construit tag_to_dof[tag] = indice compact.
    #
    # Subtilité : l'ordre des lignes dans nodeCoords n'est pas garanti
    # identique à l'ordre des tags dans nodeTags. On utilise donc
    # tag_to_node_index pour retrouver la bonne ligne de coordonnées
    # pour chaque tag.
    unique_dofs_tags = np.unique(elemNodeTags)   # tags effectivement utilisés
    num_dofs = len(unique_dofs_tags)
    max_tag  = int(np.max(nodeTags))

    all_coords = nodeCoords.reshape(-1, 3)
    tag_to_dof = np.full(max_tag + 1, -1, dtype=int)   # -1 = tag sans DDL
    dof_coords = np.zeros((num_dofs, 3), dtype=float)

    tag_to_node_index = {int(tag): i for i, tag in enumerate(nodeTags)}

    for i, tag in enumerate(unique_dofs_tags):
        tag_int           = int(tag)
        tag_to_dof[tag_int] = i
        dof_coords[i]     = all_coords[tag_to_node_index[tag_int]]

    # Points de quadrature et jacobiens - fixes et calculés une seule fois
    xi, w, N, gN    = prepare_quadrature_and_basis(elemType, order)
    jac, det, coords = get_jacobians(elemType, xi)

    # ── CL
    # OuterBoundary (côte) → Dirichlet u = 0
    #   La mer est une zone létale : les frelons qui atteignent l'eau meurent.
    # Mountains (massifs) → Neumann flux = 0 (condition naturelle)
    bnd_names = [name for name, _ in bnds]

    def get_dofs(bnd_name):
        """Retourne les DDLs d'une frontière nommée, ou un tableau vide."""
        if bnd_name in bnd_names:
            return border_dofs_from_tags(
                bnds_tags[bnd_names.index(bnd_name)], tag_to_dof
            )
        return np.array([], dtype=int)

    outer_dofs = get_dofs("OuterBoundary")
    dir_dofs   = outer_dofs.astype(int)
    dir_vals   = np.zeros(len(dir_dofs), dtype=float)   # u = 0 sur la côte


    # On approxime la côte par l'ensemble de ses nœuds du maillage.
    # Un arbre KD (structure de données spatiale) permet de répondre en
    # O(log n) à la question "quel est le nœud côtier le plus proche ?"
    # pour chacun des num_dofs nœuds du domaine.
    # C'est bien plus rapide qu'une boucle naïve en O(n²).
    coast_xy = dof_coords[outer_dofs, :2]
    coast_tree = cKDTree(coast_xy)
    dist_coast_nodal, _ = coast_tree.query(dof_coords[:, :2])

    K_nodal = np.empty(num_dofs, dtype=float)

    for i in range(num_dofs):
        x = dof_coords[i]

        #halo côtier
        d_coast = dist_coast_nodal[i]
        K_base  = (K_COAST + (d_coast / COAST_BAND) * (K_RURAL - K_COAST)
                   if d_coast < COAST_BAND else K_RURAL)

        #effet urbain (ville la plus restrictive)
        K_city = K_RURAL
        for _, cx, cy, r_hard, r_soft in CITIES:
            dist_city = math.sqrt((x[0] - cx)**2 + (x[1] - cy)**2)
            if dist_city < r_hard:
                local_K = K_URBAN
            elif dist_city < r_soft:
                t_city  = (dist_city - r_hard) / (r_soft - r_hard)
                local_K = K_URBAN + t_city * (K_RURAL - K_URBAN)
            else:
                local_K = K_RURAL
            K_city = min(K_city, local_K)

        K_local = min(K_base, K_city)

        # bonus bocager (exponentielle décroissante)
        dist_bocage = math.sqrt((x[0] - BOCAGE_CX)**2 + (x[1] - BOCAGE_CY)**2)
        K_bonus     = (K_FOREST - K_RURAL) * math.exp(-dist_bocage / 25.0)

        K_nodal[i] = K_local + K_bonus

    # Pour chaque ville, on crée un masque booléen indiquant quels nœuds
    # se trouvent dans sa zone d'influence (r_soft).
    #
    # On utilise r_soft plutôt que r_hard car certaines petites villes
    # (Corte : r_hard = 1.6 km) ont un rayon inférieur à la taille de
    # maille (~2-6 km) → aucun nœud ne tomberait dans le masque r_hard,
    # ce qui produirait des nan dans np.mean().
    city_core_masks = {}
    for name, cx, cy, r_hard, r_soft in CITIES:
        dist_city_nodal = np.array([
            math.sqrt((dof_coords[i, 0] - cx)**2 + (dof_coords[i, 1] - cy)**2)
            for i in range(num_dofs)
        ])
        city_core_masks[name] = dist_city_nodal < r_soft

    # On modélise l'introduction initiale des frelons par une 
    # gaussienne 2D centrée en (X0, Y0) au sud de l'île.
    # L'amplitude A0 et le rayon R0 définissent la taille du foyer.
    #
    # Le min(U0, K_nodal) garantit que la densité initiale ne dépasse
    # jamais la capacité locale : physiquement, on ne peut pas introduire
    # plus de frelons que l'environnement ne peut en supporter.
    #
    # Dirichlet est appliqué immédiatement : les nœuds côtiers partent à 0.
    R0, A0 = 5.0, 5.0   # rayon [km] et amplitude [ind/km²] du foyer initial
    U0 = np.array([
        A0 * math.exp(-((dof_coords[i, 0] - X0)**2 +
                         (dof_coords[i, 1] - Y0)**2) / (2 * R0**2))
        for i in range(num_dofs)
    ], dtype=float)
    U0 = np.minimum(U0, K_nodal)
    U0[dir_dofs] = 0.0


    # M ne dépend que de la géométrie → assemblée une seule fois ici.
    # M_lump (somme de chaque ligne) est utilisée dans l'IMEX pour évaluer
    # la réaction explicite nœud par nœud sans multiplication matricielle.
    M_lil  = assemble_mass(elemTags, elemNodeTags, det, w, N, tag_to_dof)
    M      = M_lil.tocsr()
    M_lump = np.array(M.sum(axis=1)).flatten()

    # Prétraitement Newton 
    #
    # preprocess_newton_data calcule une seule fois des quantités coûteuses
    # (inversions de jacobiens, gradients physiques) qui sont réutilisées
    # à chaque itération Newton de chaque pas de temps.
    #
    # Encapsulé dans try/except : si on n'utilise que l'IMEX, ce calcul
    # est inutile. S'il échoue pour une raison quelconque, newton_data
    # vaut None et l'IMEX continue de fonctionner normalement,
    try:
        from newton_solver import preprocess_newton_data
        newton_data = preprocess_newton_data(
            elemTags=elemTags, conn=elemNodeTags,
            jac=jac, det=det, xphys=coords,
            w=w, N=N, gN=gN,
            tag_to_dof=tag_to_dof, K_nodal=K_nodal
        )
    except Exception:
        newton_data = None

    # dictionnaire retourné
    return {
        # maillage brut (nécessaire pour la visualisation)
        "elemType": elemType, "nodeTags": nodeTags, "nodeCoords": nodeCoords,
        "elemTags": elemTags, "elemNodeTags": elemNodeTags,
        "bnds": bnds, "bnds_tags": bnds_tags,
        # DDLs et coordonnées
        "num_dofs": num_dofs, "tag_to_dof": tag_to_dof, "dof_coords": dof_coords,
        # quadrature et jacobiens
        "xi": xi, "w": w, "N": N, "gN": gN, "jac": jac, "det": det, "coords": coords,
        # matrices
        "M": M, "M_lump": M_lump,
        # conditions aux limites
        "dir_dofs": dir_dofs, "dir_vals": dir_vals, "outer_dofs": outer_dofs,
        # champs physiques
        "K_nodal": K_nodal, "kappa_fun": kappa_fun, "dkappa_du": dkappa_du,
        "R_GROWTH": R_GROWTH,
        # diagnostics
        "city_core_masks": city_core_masks, "dist_coast_nodal": dist_coast_nodal,
        # condition initiale
        "U0": U0,
        # données Newton (None si non disponible)
        "newton_data": newton_data,
    }


# =============================================================================
# SECTION 5 — Boucle temporelle
# =============================================================================

def run_simulation(problem, method="imex", dt=0.1, nsteps=600,
                   save_every=1, theta=1.0, live=False):
    """
    Boucle temporelle Fisher-KPP.

    À chaque pas de temps, avance la solution de tⁿ à t^{n+1} en appelant
    imex_step ou newton_solver selon la méthode choisie.

    Les snapshots (copies de U à certains pas) sont sauvegardés en mémoire
    pour générer le GIF a posteriori. Cela n'affecte pas le calcul lui-même.

    Paramètres
    ----------
    problem    : dict renvoyé par build_problem()
    method     : "imex" ou "newton"
    dt         : pas de temps [années]
    nsteps     : nombre total de pas de temps
    save_every : fréquence de sauvegarde (1 = tous les pas, 5 = un sur cinq...)
    theta      : paramètre du schéma θ (1.0 = Euler implicite)
    live       : si True, affiche la carte en temps réel pendant le calcul

    Retour
    ------
    dict contenant : times, fields, diagnostics, final_state, method, dt, nsteps
    """
    dir_dofs         = problem["dir_dofs"]
    dir_vals         = problem["dir_vals"]
    city_core_masks  = problem["city_core_masks"]
    dist_coast_nodal = problem["dist_coast_nodal"]
    U        = problem["U0"].copy()   # copie
    num_dofs = len(U)

    saved_times  = []
    saved_fields = []

    # Les clés dynamiques (f"u_mean_{name}") permettent de suivre chaque
    # ville sans dupliquer le code — la liste s'adapte à CITIES automatiquement.
    diagnostics = {
        "u_max": [],
        "invaded_fraction": [],
        "u_mean_coast":[],
        **{f"u_mean_{name}": [] for name in city_core_masks},
    }

    if live:
        fig, ax, cb = _init_live_figure()
    else:
        fig = ax = cb = None

    #boucle temporelle principale
    for step in range(nsteps):
        t = step * dt   # temps physique au début du pas courant

        # avance d'un pas de temps
        if method == "imex":
            # Diffusion implicite + réaction explicite → système linéaire
            U = imex_step(U, problem, dt, theta)

        elif method == "newton":
            # Diffusion ET réaction implicites → système non linéaire résolu
            # par Newton-Raphson (plus précis mais bien plus lent)
            U = newton_solver(
                U_init=U.copy(), U_old=U,
                M=problem["M"], dt=dt,
                newton_data=problem["newton_data"],
                kappa_fun=problem["kappa_fun"],
                dkappa_du=problem["dkappa_du"],
                r_growth=problem["R_GROWTH"],
                dirichlet_dofs=dir_dofs,
                dirichlet_vals=dir_vals
            )
            # Newton ne garantit pas u ≥ 0 → garde-fou explicite
            U = np.maximum(U, 0.0)
            U[dir_dofs] = dir_vals

        else:
            raise ValueError(f"Méthode inconnue : '{method}'. Choisir 'imex' ou 'newton'.")

        # On sauvegarde U tous les save_every pas. Le calcul continue
        # normalement entre les sauvegardes : save_every n'affecte pas
        # la physique, seulement la résolution temporelle du GIF.
        if step % save_every == 0:
            saved_times.append(t + dt)
            saved_fields.append(U.copy())

        n_inv = np.sum(U > 1.0)   # nœuds avec plus d'1 ind/km² (invasion établie)
        diagnostics["u_max"].append(np.max(U))
        diagnostics["invaded_fraction"].append(n_inv / num_dofs)
        diagnostics["u_mean_coast"].append(np.mean(U[dist_coast_nodal < 5.0]))
        for name, mask in city_core_masks.items():
            diagnostics[f"u_mean_{name}"].append(np.mean(U[mask]))

        # On ne redessine que tous les 3 pas pour limiter le coût
        # de rendu matplotlib qui peut être plus lent que le calcul
        if live and step % 3 == 0:
            cb = _update_live_figure(fig, ax, cb, problem, U, t, method)

        # Tous les 30 pas (= toutes les 3 ans avec dt=0.1) pour rester lisible.
        if step % 30 == 0 or step == nsteps - 1:
            city_stats = " | ".join(
                f"{name}={np.mean(U[mask]):.2f}"
                for name, mask in city_core_masks.items()
            )
            print(
                f"  t={t+dt:5.1f} an | u_max={np.max(U):5.1f} | "
                f"envahis={100*n_inv//num_dofs}% | {city_stats} | "
                f"côte={np.mean(U[dist_coast_nodal < 5.0]):.2f}"
            )

    if live:
        plt.ioff()
        plt.show()

    return {
        "times": np.array(saved_times),
        "fields": saved_fields,
        "diagnostics": diagnostics,
        "final_state": U.copy(),
        "method":method,
        "dt":dt,
        "nsteps":nsteps,
    }


# =============================================================================
# SECTION 6 — Visualisation
# =============================================================================

CYAN    = '#00E5FF' 
GOLD    = '#FFD700'  
LORANGE = '#FFAB40'   

# (les petits massifs ont un fontsize réduit pour éviter le chevauchement)
MASSIFS = [
    (MTN1_CX, MTN1_CY, 7.0),   # Monte Cinto — grand massif
    (MTN2_CX, MTN2_CY, 7.0),   # Monte Rotondo — grand massif
    (MTN3_CX, MTN3_CY, 4.0),   # Monte d'Oro — petit, fontsize réduit
    (MTN4_CX, MTN4_CY, 6.0),   # Monte Renoso
    (MTN5_CX, MTN5_CY, 4.0),   # Monte Incudine — petit, fontsize réduit
]


def add_overlays(ax, t_year):
    """
    Ajoute les annotations géographiques sur la carte matplotlib.

    Dessine pour chaque ville deux cercles (cœur et banlieue) et des
    labels. Ajoute aussi les labels des massifs, du bocage et le point
    marquant le foyer d'invasion initial.

    Le halo (path_effects) assure la lisibilité des textes sur le fond sombre.
    """
    theta_arc = np.linspace(0, 2 * math.pi, 300)
    halo = [pe.withStroke(linewidth=2, foreground='black')]

    for name, cx, cy, r_hard, r_soft in CITIES:
        # Cercle plein = cœur urbain strict (κ = κ_urbain)
        ax.plot(cx + r_hard * np.cos(theta_arc),
                cy + r_hard * np.sin(theta_arc),
                color=CYAN, lw=1.5, ls='-', zorder=10, alpha=0.9)
        # Cercle tirets = limite de la banlieue (fin de la transition)
        ax.plot(cx + r_soft * np.cos(theta_arc),
                cy + r_soft * np.sin(theta_arc),
                color=CYAN, lw=0.8, ls='--', zorder=10, alpha=0.6)
        ax.text(cx, cy, f'{name}\n(K={K_URBAN})',
                color=CYAN, fontsize=6.8, ha='center', va='center',
                fontweight='bold', zorder=11, path_effects=halo)
        ax.text(cx, cy - r_soft - 2.0, 'banlieues (K→50)',
                color=CYAN, fontsize=5.5, ha='center',
                zorder=11, path_effects=halo)

    for cx, cy, fs in MASSIFS:
        ax.text(cx, cy, 'Massif\n(flux=0)', color=LORANGE,
                fontsize=fs, ha='center', va='center',
                fontweight='bold', zorder=11, path_effects=halo)

    ax.text(BOCAGE_CX, BOCAGE_CY, 'Bocage\n(K=80)', color=GOLD,
            fontsize=6.5, ha='center', va='center',
            zorder=11, alpha=0.85, path_effects=halo)

    ax.plot(X0, Y0, 'x', color='white', ms=6, mew=1.5, zorder=12)
    if t_year < 15:   # on n'affiche l'annotation du foyer que les premières années
        ax.text(X0 + 5, Y0 + 5, 'Foyer\ninitial', color='white',
                fontsize=6, ha='center', zorder=11, path_effects=halo)


def make_legend(fig, ax, c_star):
    """Construit la légende de la carte avec les entrées couleur/signification."""
    patches = [
        mpatches.Patch(facecolor='black', edgecolor='white', lw=0.5,
                       label='Aucune invasion (u ≈ 0)'),
        mpatches.Patch(facecolor='#7b2d8b', edgecolor='none',
                       label="Front d'invasion actif"),
        mpatches.Patch(facecolor='#f4a261', edgecolor='none',
                       label='Population établie'),
        mpatches.Patch(facecolor='#ffff5e', edgecolor='none',
                       label=f'Saturation K (campagne={K_RURAL:.0f} ind/km²)'),
        mpatches.Patch(facecolor='black', edgecolor='white', lw=0.5,
                       label='Mer : u = 0'),
        mpatches.Patch(facecolor='white', edgecolor='gray', lw=0.5,
                       label='Massifs : flux = 0'),
        mpatches.Patch(facecolor='black', edgecolor=CYAN, lw=1.5,
                       label=f'Villes (K={K_URBAN})'),
    ]
    leg = ax.legend(
        handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.18),
        ncol=3, frameon=True, framealpha=0.85,
        facecolor='#1a1a2e', edgecolor='#444466',
        labelcolor='white', fontsize=7.5,
        title=f'c* = {c_star:.1f} km/an   |   Modèle Fisher-KPP',
        title_fontsize=8,
    )
    leg.get_title().set_color('#aaaacc')
    return leg


def _init_live_figure():
    """
    Crée la fenêtre matplotlib pour l'affichage en temps réel.
    plt.ion() active le mode interactif : plt.pause() met à jour la fenêtre
    sans bloquer le programme.
    """
    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 10))
    fig.patch.set_facecolor('#0d0d1a')
    ax.set_facecolor('#0d0d1a')
    return fig, ax, None


def _update_live_figure(fig, ax, cb, problem, U, t, method):
    """
    Redessine la carte à l'instant t avec la solution U courante.

    La colorbar cb est créée au premier appel et réutilisée ensuite
    (cb_holder pattern) pour éviter de multiplier les colorbars.
    plt.pause(0.02) force le rendu matplotlib et cède brièvement la main
    à l'OS pour que la fenêtre reste réactive.
    """
    c_star = 2.0 * math.sqrt(KAPPA_RURAL * R_GROWTH)
    ax.clear()
    ax.set_facecolor('#0d0d1a')

    contour = plot_fe_solution_2d(
        elemNodeTags=problem["elemNodeTags"], nodeTags=problem["nodeTags"],
        nodeCoords=problem["nodeCoords"], U=U, tag_to_dof=problem["tag_to_dof"],
        show_mesh=False, ax=ax, vmin=0.0, vmax=K_FOREST, cmap='plasma'
    )
    add_overlays(ax, t)
    make_legend(fig, ax, c_star)

    ax.set_title(
        f'Invasion du Frelon Asiatique — Fisher-KPP (Corse)\n'
        f't = {t:.1f} an  |  c* = {c_star:.1f} km/an  |  méthode : {method}',
        color='white', fontsize=11, pad=10
    )
    ax.set_xlabel('x [km]', color='#aaaacc')
    ax.set_ylabel('y [km]', color='#aaaacc')
    ax.tick_params(colors='#aaaacc')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')
    ax.axis('equal')

    if cb is None:
        cb = fig.colorbar(contour, ax=ax, label='Densité u [ind/km²]',
                          pad=0.02, fraction=0.03)
        cb.ax.yaxis.label.set_color('white')
        cb.ax.tick_params(colors='white')

    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig.canvas.draw()
    plt.pause(0.02)
    return cb


def save_results_animation(problem, results, output_file="simulation.gif",
                            stride=1, fps=10):
    """
    Génère un GIF animé à partir des snapshots sauvegardés par run_simulation.

    Contrairement au mode --live (affichage pendant le calcul), cette fonction
    est appelée APRÈS la simulation : elle relit les champs sauvegardés et
    génère le GIF à la vitesse souhaitée via fps.

    Paramètres
    ----------
    problem     : dict renvoyé par build_problem()
    results     : dict renvoyé par run_simulation()
    output_file : nom du fichier de sortie (.gif)
    stride      : ne conserve qu'une frame sur `stride` (réduit la taille du GIF)
    fps         : images par seconde — plus grand = animation plus rapide
    """
    times  = results["times"][::stride]
    fields = results["fields"][::stride]
    c_star = 2.0 * math.sqrt(KAPPA_RURAL * R_GROWTH)
    method = results["method"]

    fig, ax = plt.subplots(figsize=(9, 10))
    fig.patch.set_facecolor('#0d0d1a')
    ax.set_facecolor('#0d0d1a')
    cb_holder = {"obj": None}   # dict mutable pour contourner la closure Python

    def update(frame_idx):
        """Callback appelé par FuncAnimation pour chaque frame."""
        t = times[frame_idx]
        U = fields[frame_idx]
        ax.clear()
        ax.set_facecolor('#0d0d1a')

        contour = plot_fe_solution_2d(
            elemNodeTags=problem["elemNodeTags"], nodeTags=problem["nodeTags"],
            nodeCoords=problem["nodeCoords"], U=U, tag_to_dof=problem["tag_to_dof"],
            show_mesh=False, ax=ax, vmin=0.0, vmax=K_FOREST, cmap='plasma'
        )
        add_overlays(ax, t)
        make_legend(fig, ax, c_star)

        ax.set_title(
            f'Invasion du Frelon Asiatique — Fisher-KPP (Corse)\n'
            f't = {t:.1f} an  |  c* = {c_star:.1f} km/an  |  méthode : {method}',
            color='white', fontsize=11, pad=10
        )
        ax.set_xlabel('x [km]', color='#aaaacc')
        ax.set_ylabel('y [km]', color='#aaaacc')
        ax.tick_params(colors='#aaaacc')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')
        ax.axis('equal')

        if cb_holder["obj"] is None:
            cb = fig.colorbar(contour, ax=ax, label='Densité u [ind/km²]',
                              pad=0.02, fraction=0.03)
            cb.ax.yaxis.label.set_color('white')
            cb.ax.tick_params(colors='white')
            cb_holder["obj"] = cb

        fig.tight_layout(rect=[0, 0.12, 1, 1])

    anim = FuncAnimation(fig, update, frames=len(times), repeat=False)
    anim.save(output_file, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Animation sauvegardée : {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Fisher-KPP 2D — Invasion du frelon asiatique (Vespa velutina)")
    parser.add_argument("-order", type=int, default=1, help="Ordre polynomial des éléments (défaut : 1)")
    parser.add_argument("--theta",type=float, default=1.0, help="θ du schéma (1.0=Euler implicite, 0.5=Crank-Nicolson)")
    parser.add_argument("--dt", type=float, default=0.1, help="Pas de temps [années] (défaut : 0.1)")
    parser.add_argument("--nsteps", type=int, default=600, help="Nombre de pas de temps (défaut : 600 → 60 ans)")
    parser.add_argument("--method",type=str, default="imex", choices=["imex", "newton"], help="Méthode temporelle (défaut : imex)")
    parser.add_argument("--save_every", type=int, default=5, help="Sauvegarde 1 snapshot tous les N pas (défaut : 5)")
    parser.add_argument("--live", action="store_true", help="Affichage en temps réel pendant le calcul")
    parser.add_argument("--no_visu", action="store_true", help="Ne génère pas de GIF à la fin")
    args = parser.parse_args()

    #préparation (maillage, matrices, K_nodal, U0, ...)
    problem = build_problem(order=args.order)

    # Résumé des paramètres avant de lancer
    c_star  = 2.0 * math.sqrt(KAPPA_RURAL * R_GROWTH)
    T_total = args.dt * args.nsteps

    print(f"\n{'═' * 62}")
    print("  Fisher-KPP — Vespa velutina (Corse)")
    print(f"  Villes : {', '.join(name for name, *_ in CITIES)}")
    print(f"  κ(u,x) = κ_base(x) / (1 + {ALPHA_KAPPA}·u)")
    print(f"  κ rural={KAPPA_RURAL} | κ urbain={KAPPA_URBAN} km²/an")
    print(f"  r={R_GROWTH} an⁻¹  |  c* = {c_star:.2f} km/an")
    print(f"  K côte={K_COAST} → rural={K_RURAL} → bocage={K_FOREST} ind/km²")
    print(f"  méthode : {args.method}  |  θ = {args.theta}")
    print(f"  dt={args.dt} an  |  {args.nsteps} pas  |  T={T_total:.0f} ans")
    print(f"  DDLs={problem['num_dofs']}  |  stabilité : dt·r={args.dt*R_GROWTH:.2f} < 1 ✓")
    print(f"{'═' * 62}\n")

    results = run_simulation(
        problem,
        method=args.method,
        dt=args.dt,
        nsteps=args.nsteps,
        save_every=args.save_every,
        theta=args.theta,
        live=args.live,
    )

    print(f"\nSimulation terminée.")
    print(f"  Snapshots sauvegardés : {len(results['fields'])}")
    print(f"  u_max final           = {results['diagnostics']['u_max'][-1]:.3f} ind/km²")
    print(f"  Fraction envahie      = {results['diagnostics']['invaded_fraction'][-1]:.1%}")

    #post-traitement (génération du GIF)
    if not args.no_visu:
        save_results_animation(problem, results,output_file="simulation.gif", stride=1, fps=10)

    gmsh_finalize()
    return problem, results


if __name__ == "__main__":
    main()