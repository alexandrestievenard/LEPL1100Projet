import json
from pathlib import Path

from shapely.geometry import shape, MultiPolygon, Polygon
from shapely.ops import transform
from pyproj import Transformer


INPUT_FILE = "region-corse.geojson"
OUTPUT_FILE = "corse_coords.py"

# tolérance de simplification en mètres
SIMPLIFY_TOL_METERS = 800.0


def load_main_polygon(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    geom = shape(data["geometry"])

    if isinstance(geom, MultiPolygon):
        # on garde l'île principale
        poly = max(geom.geoms, key=lambda g: g.area)
    elif isinstance(geom, Polygon):
        poly = geom
    else:
        raise ValueError(f"Géométrie non supportée : {geom.geom_type}")

    return poly


def project_polygon(poly):
    # WGS84 (lon/lat) -> Lambert-93 (mètres)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    return transform(transformer.transform, poly)


def polygon_to_local_km(poly):
    coords = list(poly.exterior.coords)

    # enlever le dernier point s'il répète le premier
    if coords[0] == coords[-1]:
        coords = coords[:-1]

    xs = [x for x, y in coords]
    ys = [y for x, y in coords]

    xmin = min(xs)
    ymin = min(ys)

    # translation pour avoir un domaine local commençant près de (0,0)
    coords_km = [((x - xmin) / 1000.0, (y - ymin) / 1000.0) for x, y in coords]
    return coords_km, xmin, ymin


def main():
    poly = load_main_polygon(INPUT_FILE)
    poly_proj = project_polygon(poly)

    # simplification géométrique
    poly_simpl = poly_proj.simplify(SIMPLIFY_TOL_METERS, preserve_topology=True)

    coords_km, xmin, ymin = polygon_to_local_km(poly_simpl)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("# Coordonnées simplifiées de la Corse en km\n")
        f.write(f"# origine projetée (m) : xmin={xmin}, ymin={ymin}\n")
        f.write("CORSE_XY = [\n")
        for x, y in coords_km:
            f.write(f"    ({x:.6f}, {y:.6f}),\n")
        f.write("]\n")

    print(f"Fichier écrit : {OUTPUT_FILE}")
    print(f"Nombre de points : {len(coords_km)}")


if __name__ == "__main__":
    main()