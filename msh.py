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
# 5. Massifs montagneux corses (approximation grossière)
# -----------------------------------------------------------------------------
# Ajustables après un premier essai visuel.
#
# M1 : grand massif centre-nord / nord-ouest intérieur
# M2 : petit massif centre-sud
# M3 : massif sud-est intérieur

mtn1_xy = [
    (18.75, 110.0),
    (22.0, 118.75),
    (34.0, 127.0),
    (41.25, 128.0),
    (47.5, 114.0),
    (50.0, 94.0),
    (44.0, 83.0),
    (34.0, 83.0),
    (22.5, 93.25),
]

mtn2_xy = [
    (53.0, 85.0),
    (56.0, 85.0),
    (54.0, 68.75),
    (49.0, 63.0),
    (44.25, 62.5),
    (39.25, 66.0),
    (40.0, 75.0),
    (49.2, 81.0),
]

mtn3_xy = [
    (57.5, 80.0),
    (62.5, 80.5),
    (65.0, 62.0),
    (64.0, 50.0),
    (54.5, 43.0),
    (45.0, 44.25),
    (44.5, 50.5),
    (57.5, 71.0),
]

mtn1_pt_tags, mtn1_line_tags, mtn1_loop = make_polygon_loop(mtn1_xy, CL_MTN)
mtn2_pt_tags, mtn2_line_tags, mtn2_loop = make_polygon_loop(mtn2_xy, CL_MTN)
mtn3_pt_tags, mtn3_line_tags, mtn3_loop = make_polygon_loop(mtn3_xy, CL_MTN)


# -----------------------------------------------------------------------------
# 6. Domaine final : soustraction des massifs
# -----------------------------------------------------------------------------
outer_surf = gmsh.model.occ.addPlaneSurface([outer_loop])

final_surf_tag = gmsh.model.occ.addPlaneSurface([
    outer_loop,
    mtn1_loop,
    mtn2_loop,
    mtn3_loop
])

gmsh.model.occ.synchronize()

# -----------------------------------------------------------------------------
# 8. Groupes physiques
# -----------------------------------------------------------------------------
outer_curves = outer_line_tags
mtn_curves = mtn1_line_tags + mtn2_line_tags + mtn3_line_tags

gmsh.model.addPhysicalGroup(2, [final_surf_tag], tag=1)
gmsh.model.setPhysicalName(2, 1, "Domain")

gmsh.model.addPhysicalGroup(1, outer_curves, tag=10)
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
gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", outer_curves)

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