# =============================================================================
# msh.py — Génération du maillage 2D pour l'invasion du frelon asiatique
# =============================================================================
#
# Ce script génère le fichier "invasion_map.msh" utilisé par main_diffusion_2d.py.
# Il doit être exécuté en premier : python msh.py
#
# STRUCTURE DU DOMAINE :
#   Ω = Ω_ext \ (Ω_lac ∪ Ω_montagne)
#
#   - Frontière extérieure (OuterBoundary) : polygone 8 sommets ~100×100 km
#     → Condition de Neumann flux=0 (domaine fermé, les frelons ne sortent pas)
#
#   - Lac de la Garenne (Lake) : ellipse soustraite → trou dans le maillage
#     → Condition de Dirichlet u=0 (barrière létale, les frelons s'y noient)
#
#   - Massif des Crêtes (Mountains) : polygone soustrait → trou dans le maillage
#     → Condition de Neumann flux=0 (obstacle infranchissable, les frelons se
#       réfléchissent et s'accumulent contre la paroi jusqu'à saturation K)
#
#   - Métropole Centrale : PAS un trou — elle est incluse dans le domaine.
#     Son effet hostile est modélisé via κ(x) réduit et K(x) faible dans main.py.
#     Ne crée pas de condition aux limites ici.
#
# RAFFINEMENT DU MAILLAGE :
#   Maille fine (~1.5 km) au coin SW (point d'entrée de l'invasion),
#   maille plus grossière (~6 km) au loin.
# =============================================================================

import gmsh
import math


# -----------------------------------------------------------------------------
# SECTION 1 — Initialisation de Gmsh
# -----------------------------------------------------------------------------
gmsh.initialize()
gmsh.model.add("invasion_frelon_asiatique")


# -----------------------------------------------------------------------------
# SECTION 2 — Tailles de maille caractéristiques (en km)
# -----------------------------------------------------------------------------
CL_OUTER    = 6.0    # Maille grossière sur la frontière extérieure
CL_LAKE     = 2.0    # Maille fine autour du lac (bords courbes)
CL_MTN      = 2.5    # Maille fine autour du massif (bords anguleux)
CL_INVASION = 1.5    # Maille très fine au coin SW (foyer d'invasion)


# -----------------------------------------------------------------------------
# SECTION 3 — Frontière extérieure du domaine (polygone irrégulier 8 sommets)
#
# Ce contour représente une région géographique fictive. L'irrégularité
# (côte sinueuse, caps, col SW) valide la robustesse du maillage et rend
# le domaine plus réaliste qu'un simple rectangle.
# Le coin SW (0,0) est le point d'entrée géographique de l'invasion.
# -----------------------------------------------------------------------------
outer_xy = [
    (  0.0,   0.0),   # col SW — point d'entrée de l'invasion
    ( 40.0,  -4.0),   # plaine côtière sud
    ( 80.0,  -2.0),   # plaine méridionale
    (100.0,   0.0),   # cap SE (pointe maritime)
    (104.0,  50.0),   # côte orientale
    (100.0, 100.0),   # cap NE
    ( 50.0, 104.0),   # frontière nord (ligne de crête)
    (  0.0, 100.0),   # col NW
]

outer_pt_tags = [
    gmsh.model.occ.addPoint(x, y, 0, CL_OUTER) for x, y in outer_xy
]
n_outer = len(outer_pt_tags)

outer_line_tags = [
    gmsh.model.occ.addLine(outer_pt_tags[i], outer_pt_tags[(i + 1) % n_outer])
    for i in range(n_outer)
]
outer_loop = gmsh.model.occ.addCurveLoop(outer_line_tags)


# -----------------------------------------------------------------------------
# SECTION 4 — Lac de la Garenne (ellipse → trou dans le maillage)
#
# Demi-axes (Rx=13, Ry=7) km, centré en (22, 68) km.
# Sera soustrait du domaine : aucun nœud à l'intérieur.
# Condition aux limites associée dans main.py : Dirichlet u=0.
# -----------------------------------------------------------------------------
LAKE_CX, LAKE_CY = 22.0, 68.0
LAKE_RX, LAKE_RY = 13.0,  7.0

lake_disk = gmsh.model.occ.addDisk(LAKE_CX, LAKE_CY, 0, LAKE_RX, LAKE_RY)


# -----------------------------------------------------------------------------
# SECTION 5 — Massif des Crêtes (polygone 9 sommets → trou dans le maillage)
#
# Représente une chaîne montagneuse infranchissable.
# Sera soustrait du domaine : aucun nœud à l'intérieur.
# Condition aux limites associée dans main.py : Neumann flux=0 (naturel).
# -----------------------------------------------------------------------------
MTN_CX, MTN_CY = 76.0, 70.0   # centroïde approximatif (utilisé pour la classification)

mtn_xy = [
    (62.0, 60.0), (70.0, 55.0), (78.0, 56.0),
    (86.0, 62.0), (90.0, 71.0), (87.0, 81.0),
    (79.0, 84.0), (70.0, 81.0), (62.0, 72.0),
]
mtn_pt_tags = [gmsh.model.occ.addPoint(x, y, 0, CL_MTN) for x, y in mtn_xy]
nm = len(mtn_pt_tags)
mtn_line_tags = [
    gmsh.model.occ.addLine(mtn_pt_tags[i], mtn_pt_tags[(i + 1) % nm])
    for i in range(nm)
]
mtn_loop = gmsh.model.occ.addCurveLoop(mtn_line_tags)
mtn_surf = gmsh.model.occ.addPlaneSurface([mtn_loop])


# -----------------------------------------------------------------------------
# SECTION 6 — Soustraction booléenne : Ω = Ω_ext \ (Ω_lac ∪ Ω_montagne)
#
# NB : La Métropole Centrale n'est PAS soustraite ici.
#      Elle reste dans le domaine de calcul et son effet est modélisé
#      via les champs κ(x) et K(x) dans main_diffusion_2d.py.
# -----------------------------------------------------------------------------
outer_surf = gmsh.model.occ.addPlaneSurface([outer_loop])

out, _ = gmsh.model.occ.cut(
    [(2, outer_surf)],
    [(2, lake_disk), (2, mtn_surf)],
    removeObject=True,
    removeTool=True,
)
gmsh.model.occ.synchronize()

# Vérification : il doit rester exactement 1 surface après la soustraction
domains = gmsh.model.getEntities(2)
assert len(domains) == 1, (
    f"ERREUR : {len(domains)} surface(s) trouvée(s) après la coupe, attendu 1.\n"
    "Vérifiez que les obstacles sont bien à l'intérieur du domaine extérieur."
)
final_surf_tag = domains[0][1]


# -----------------------------------------------------------------------------
# SECTION 7 — Classification des courbes frontières
#
# Après la soustraction booléenne, les tags Gmsh des courbes peuvent changer.
# On classe donc chaque courbe par sa position géographique (boîte englobante),
# ce qui est robuste face aux renumérotations internes de Gmsh.
#
# Logique :
#   - Courbe dont la bbox est dans la zone du lac → Lake
#   - Courbe dont la bbox est dans la zone du massif → Mountains
#   - Tout le reste → OuterBoundary
# -----------------------------------------------------------------------------
all_bnd = gmsh.model.getBoundary([(2, final_surf_tag)], oriented=False)
all_bnd_set = {abs(tag) for (_, tag) in all_bnd}

outer_curves = []
lake_curves  = []
mtn_curves   = []

for ctag in all_bnd_set:
    xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, ctag)

    # Zone lac : X ∈ [9, 35] km, Y ∈ [61, 75] km (ellipse ± marge)
    if xmin >= 8.0 and xmax <= 36.0 and ymin >= 60.0 and ymax <= 76.0:
        lake_curves.append(ctag)
    # Zone massif : X ∈ [61, 91] km, Y ∈ [54, 85] km (polygone ± marge)
    elif xmin >= 61.0 and xmax <= 91.0 and ymin >= 54.0 and ymax <= 85.0:
        mtn_curves.append(ctag)
    else:
        outer_curves.append(ctag)

print("─" * 50)
print(f"  OuterBoundary : {len(outer_curves)} courbe(s)")
print(f"  Lake          : {len(lake_curves)} courbe(s)")
print(f"  Mountains     : {len(mtn_curves)} courbe(s)")
print(f"  City          : dans le domaine (pas de trou)")
print("─" * 50)


# -----------------------------------------------------------------------------
# SECTION 8 — Groupes physiques
#
# Les groupes physiques sont les noms que main_diffusion_2d.py utilisera
# pour retrouver les nœuds de chaque frontière via open_2d_mesh().
# Tags entiers arbitraires mais uniques (1, 10, 11, 12).
# -----------------------------------------------------------------------------
gmsh.model.addPhysicalGroup(2, [final_surf_tag], tag=1)
gmsh.model.setPhysicalName(2, 1, "Domain")

gmsh.model.addPhysicalGroup(1, outer_curves, tag=10)
gmsh.model.setPhysicalName(1, 10, "OuterBoundary")

if lake_curves:
    gmsh.model.addPhysicalGroup(1, lake_curves, tag=11)
    gmsh.model.setPhysicalName(1, 11, "Lake")

if mtn_curves:
    gmsh.model.addPhysicalGroup(1, mtn_curves, tag=12)
    gmsh.model.setPhysicalName(1, 12, "Mountains")


# -----------------------------------------------------------------------------
# SECTION 9 — Paramètres de maillage et raffinement adaptatif
#
# On utilise un champ "Distance + Threshold" pour raffiner la maille
# uniquement près du coin SW (foyer d'invasion), sans alourdir le reste.
#
# Résultat : maille ~1.5 km dans un rayon de 8 km autour de (0,0),
#            transition vers ~6 km au-delà de 30 km.
# -----------------------------------------------------------------------------
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 1.2)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 7.0)
gmsh.option.setNumber("Mesh.Algorithm", 6)   # Frontal-Delaunay : qualité optimale

# Désactiver les heuristiques de taille automatique de Gmsh
# (on utilise uniquement notre champ de background)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

# Champ de distance au coin SW (tag du point = outer_pt_tags[0])
f_dist = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(f_dist, "PointsList", [outer_pt_tags[0]])

# Champ seuil : interpolation linéaire de SizeMin à SizeMax entre DistMin et DistMax
f_thresh = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(f_thresh, "InField",  f_dist)
gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin",  CL_INVASION)  # fin (< 8 km du SW)
gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax",  CL_OUTER)     # grossier (> 30 km)
gmsh.model.mesh.field.setNumber(f_thresh, "DistMin",  8.0)
gmsh.model.mesh.field.setNumber(f_thresh, "DistMax",  30.0)

gmsh.model.mesh.field.setAsBackgroundMesh(f_thresh)


# -----------------------------------------------------------------------------
# SECTION 10 — Génération, optimisation et export
# -----------------------------------------------------------------------------
gmsh.model.mesh.generate(2)
gmsh.model.mesh.optimize("Netgen")   # Post-optimisation : améliore les angles des triangles
gmsh.write("invasion_map.msh")

print("✓ Maillage généré : invasion_map.msh")
print(f"  Nœuds    : {len(gmsh.model.mesh.getNodes()[0])}")
print(f"  Éléments : {len(gmsh.model.mesh.getElementsByType(gmsh.model.mesh.getElementType('triangle', 1))[0])}")

gmsh.finalize()