# =============================================================================
# msh.py — Maillage 2D de la Corse avec massifs montagneux internes
# =============================================================================
#
# Domaine :
#   Ω = Corse \ (massifs montagneux)
#
# Groupes physiques :
#   - OuterBoundary : côte corse
#   - Mountains     : obstacles internes (Neumann homogène dans main)
#   - Domain        : surface de calcul
#
# Raffinement :
#   - plus fin près de toute la côte
#   - plus grossier à l’intérieur
#   - sans zone manuelle locale
# =============================================================================

import gmsh
from corse_coords import CORSE_XY


# -----------------------------------------------------------------------------
# 1. Initialisation
# -----------------------------------------------------------------------------
gmsh.initialize()
gmsh.model.add("corse_frelon_asiatique")


# -----------------------------------------------------------------------------
# 2. Paramètres de maille
# -----------------------------------------------------------------------------
CL_OUTER = 6.0
CL_MTN = 2.0

CL_BOUNDARY = 2.0   # taille près de toute la côte
CL_INTERIOR = 6.0   # taille vers l’intérieur


# -----------------------------------------------------------------------------
# 3. Fonction utilitaire : polygone -> surface OCC
# -----------------------------------------------------------------------------
def make_polygon_loop(xy, cl):
    pt_tags = [gmsh.model.occ.addPoint(x, y, 0, cl) for x, y in xy]
    line_tags = [
        gmsh.model.occ.addLine(pt_tags[i], pt_tags[(i + 1) % len(pt_tags)])
        for i in range(len(pt_tags))
    ]
    loop = gmsh.model.occ.addCurveLoop(line_tags)
    return pt_tags, line_tags, loop


# -----------------------------------------------------------------------------
# 4. Frontière extérieure : Corse
# -----------------------------------------------------------------------------
outer_xy = CORSE_XY

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
# 5. Massifs montagneux (5 massifs distincts)
# -----------------------------------------------------------------------------

# =========================================================
# 1. Monte Cinto (Nord-Ouest) - Inchangé
# =========================================================
mtn_cinto_xy = [
    (31.0, 120.0), (31.5, 121.5), (33.0, 122.5), (35.0, 125.0),
    (35.5, 127.5), (37.0, 128.2), (39.0, 128.8), (40.5, 128.9), (42.0, 128.5), (41.8, 127.0),
    (40.5, 126.5), (40.0, 125.0), (41.5, 124.7), (42.5, 125.0), 
    (43.0, 125.4), (44.3, 125.0), (44.0, 124.5), (43.2, 123.5), (41.5, 122.3), (40.5, 122.5), (39.5, 121.5), (39.4, 120.0), (40.5, 120.1),
    (41.7, 120.5), (43.5, 121.0), (45.5, 121.5), (46.2, 121.6), (47.5, 121.5), (48.5, 120.0), (49.5, 118.0),
    (49.0, 116.2), (48.5, 113.9), (47.5, 112.5), (46.0, 110.8),
    (44.0, 109.9), (42.0, 108.0), (40.0, 105.8), (38.0, 105.0),
    (36.0, 104.5), (34.0, 103.5), (32.0, 103.0), (30.0, 103.0),
    (28.0, 103.5), (26.8, 104.5), (26.8, 106.7), (27.2, 109.0), (27.7, 112.5), (28.5, 114.5), (29.0, 116.5),
    (30.0, 118.5)
]

# =========================================================
# 2. Monte Rotondo (Centre-Nord) - Inchangé
# =========================================================
mtn_rotondo_xy = [
    (38.0, 101.0), (42.0, 102.0), (46.0, 102.5), (49.0, 104.5),
    (52.0, 105.5), (54.5, 107.2), (56.0, 107.0), (56.5, 105.5), (55.5, 103.5), (54.0, 102.0), (52.0, 100.0), (50.0, 99.0), (50.0, 98.0), (51.0, 97.6), (52.5, 96.5), (52.1, 95.2), (51.2, 94.5),
    (51.0, 94.0), (52.5, 93.5), (53.5, 94.5), (54.5, 96.0), (55.7, 96.2), (56.1, 94.2), (55.8, 92.5), (55.5, 90.5), 
    (54.0, 89.0), (50.0, 88.0), (45.0, 88.5), (42.5, 89.0), (40.0, 91.0),
    (38.0, 91.5), (36.0, 92.9), (35.0, 93.8),
    (33.0, 95.0), (33.0, 98.0), (35.0, 100.0)
]

# =========================================================
# 3. Monte d'Oro (Centre) - Inchangé
# =========================================================
mtn_doro_xy = [
    (43.5, 80.0), (45.0, 81.0), (48.0, 82.0), (52.0, 81.0),
    (54.5, 79.0), (54.0, 76.0),
    (50.0, 76.0), (46.0, 75.0), (43.0, 77.0)
]

# =========================================================
# 4. Monte Renoso (Centre-Sud) - CORRIGÉ
# =========================================================
mtn_renoso_xy = [
    (43.0, 71.5),  # 0 - coin haut-gauche
    (46.0, 72.0),  # 1 - sommet nord
    (49.0, 72.8),  # 2 - haut-centre
    (52.0, 73.0),  # 3 - haut-droite
    (55.0, 73.5),  # 4 - droite haute
    (56.2, 75.0),
    (58.0, 76.0),
    (60.0, 75.5),
    (59.5, 74.0),
    (58.2, 73.0),
    (57.5, 72.0),  
    (57.2, 69.5),
    (57.0, 66.0),  
    (55.0, 63.5),  
    (52.0, 62.0),  
    (49.0, 61.5),  
    (46.0, 60.5),  
    (44.5, 59.2),
    (43.0, 57.5),
    (42.0, 57.0),
    (41.0, 58.1),
    (41.5, 59.0),
    (42.5, 60.0), 
    (44.0, 61.0),
    (45.5, 63.2),
    (45.7, 65.0),
    (44.8, 67.0),
    (43.5, 67.5),
    (41.0, 68.3),
    (40.5, 69.0),
    (41.0, 71.0),

]

# =========================================================
# 5. Monte Incudine (Sud) - CORRIGÉ ET COMPACTÉ
# Vertically compressed and further North (min Y ~38.5)
# =========================================================
mtn_incudine_xy = [
    (51.0, 56.0),  
    (48.5, 53.0),
    (45.5, 50.5),
    (43.5, 49.5),
    (41.5, 49.0),
    (39.0, 47.8),
    (38.5, 47.3),
    (38.0, 46.7),
    (38.2, 45.5), 
    (39.2, 44.3),
    (40.0, 44.5),
    (42.0, 45.0),
    (44.0, 46.5),
    (45.0, 47.0),
    (46.5, 47.5),
    (48.0, 46.5),      
    (47.0, 45.0),
    (46.8, 44.0),
    (48.0, 43.0),
    (49.5, 42.5),
    (50.0, 43.0),
    (52.5, 43.5),
    (53.5, 44.0),
    (54.8, 46.0),
    (55.8, 47.0), 
    (56.9, 48.5),
    (56.5, 50.5),
    (55.0, 52.0),
    (54.0, 55.8),
    (55.0, 58.5),
    (57.5, 61.0),    
    (59.0, 63.0),
    (60.5, 64.7),
    (61.0, 66.5),
    (60.0, 66.3),
    (58.0, 64.5),
    (56.2, 62.0),
    (54.5, 60.5),
    (52.5, 58.0),
]

mtn_cinto_pt, mtn_cinto_ln, mtn_cinto_loop       = make_polygon_loop(mtn_cinto_xy, CL_MTN)
mtn_rotondo_pt, mtn_rotondo_ln, mtn_rotondo_loop  = make_polygon_loop(mtn_rotondo_xy, CL_MTN)
mtn_doro_pt, mtn_doro_ln, mtn_doro_loop          = make_polygon_loop(mtn_doro_xy, CL_MTN)
mtn_renoso_pt, mtn_renoso_ln, mtn_renoso_loop    = make_polygon_loop(mtn_renoso_xy, CL_MTN)
mtn_incudine_pt, mtn_incudine_ln, mtn_incudine_loop = make_polygon_loop(mtn_incudine_xy, CL_MTN)


# -----------------------------------------------------------------------------
# 6. Domaine final
# -----------------------------------------------------------------------------
final_surf_tag = gmsh.model.occ.addPlaneSurface([
    outer_loop,
    mtn_cinto_loop,
    mtn_rotondo_loop,
    mtn_doro_loop,
    mtn_renoso_loop,
    mtn_incudine_loop,
])

gmsh.model.occ.synchronize()

# -----------------------------------------------------------------------------
# 8. Groupes physiques
# -----------------------------------------------------------------------------
mtn_curves = (mtn_cinto_ln + mtn_rotondo_ln + mtn_doro_ln +
              mtn_renoso_ln + mtn_incudine_ln)

gmsh.model.addPhysicalGroup(2, [final_surf_tag], tag=1)
gmsh.model.setPhysicalName(2, 1, "Domain")

gmsh.model.addPhysicalGroup(1, outer_line_tags, tag=10)
gmsh.model.setPhysicalName(1, 10, "OuterBoundary")

gmsh.model.addPhysicalGroup(1, mtn_curves, tag=12)
gmsh.model.setPhysicalName(1, 12, "Mountains")

# -----------------------------------------------------------------------------
# 9. Paramètres de maillage
# -----------------------------------------------------------------------------
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 1.5)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 7.0)
gmsh.option.setNumber("Mesh.Algorithm", 6)

gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)

# Raffinement près de toute la côte
f_dist = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", outer_line_tags)  

f_thresh = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", CL_BOUNDARY)
gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", CL_INTERIOR)
gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", 0.0)
gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", 15.0)

gmsh.model.mesh.field.setAsBackgroundMesh(f_thresh)


# -----------------------------------------------------------------------------
# 10. Génération et export
# -----------------------------------------------------------------------------
gmsh.model.mesh.generate(2)
gmsh.model.mesh.optimize("Netgen")
gmsh.write("invasion_map.msh")

print("✓ Maillage généré : invasion_map.msh")
print(f"  Nœuds    : {len(gmsh.model.mesh.getNodes()[0])}")
print(f"  Éléments : {len(gmsh.model.mesh.getElementsByType(gmsh.model.mesh.getElementType('triangle', 1))[0])}")

gmsh.finalize()