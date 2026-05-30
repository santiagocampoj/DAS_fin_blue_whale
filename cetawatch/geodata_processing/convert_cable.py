#!/usr/bin/env python3
"""
convert_cable.py
----------------
Convierte los ficheros de geometria del cable del OOI (canal/lat/lon/prof)
en los ficheros que usa el dashboard CetaWatch. Procesa uno o varios cables.

Para cada cable genera:
  1) cable_<label>.geojson        -> linea para dibujar en Leaflet (submuestreada)
  2) channel_coords_<label>.json  -> mapeo COMPLETO canal -> [lon, lat] + dc,
                                     para convertir metros->canal->coordenada.

Formato de entrada (4 columnas, separadas por espacios):
    canal   latitud   longitud   profundidad

Ficheros 2021 (OptaSense, 2 m), descargables del servidor del OOI:
    http://piweb.ooirsn.uw.edu/das/processed/metadata/Geometry/\
OOI_RCA_DAS_channel_location_with_depth/north_DAS_latlondepth.txt
    .../south_DAS_latlondepth.txt

dc (metros por canal): el espaciado nominal es 2 m, pero el geografico real
medido en los CSV de detecciones es 2.0417 m/canal. Se puede sobreescribir
con --dc. Se guarda en el JSON para que el JS convierta metros -> canal.

Uso:
    python convert_cable.py north_DAS_latlondepth.txt south_DAS_latlondepth.txt \
        --outdir data --step 25 --dc 2.0417

La etiqueta (north/south) se deduce del nombre del fichero.
"""

import argparse
import json
import os
import numpy as np


def label_from_filename(path):
    name = os.path.basename(path).lower()
    if "north" in name:
        return "north"
    if "south" in name:
        return "south"
    # si no se reconoce, usa el nombre sin extension
    return os.path.splitext(os.path.basename(path))[0]


def load_cable(txt_path):
    """Carga el .txt del OOI. Devuelve (canal, lat, lon, profundidad)."""
    data = np.genfromtxt(txt_path)
    chan = data[:, 0].astype(int)
    lat = data[:, 1]
    lon = data[:, 2]
    dep = data[:, 3]
    return chan, lat, lon, dep


def build_geojson(label, chan, lat, lon, step):
    """LineString del trazado del cable, submuestreado 1 de cada `step` puntos."""
    n = len(chan)
    coords = [[round(float(lon[i]), 6), round(float(lat[i]), 6)]
              for i in range(0, n, step)]
    last = [round(float(lon[-1]), 6), round(float(lat[-1]), 6)]
    if coords[-1] != last:
        coords.append(last)

    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {
                "name": f"OOI RCA {label.capitalize()} Cable",
                "cable": label,
                "channel_min": int(chan.min()),
                "channel_max": int(chan.max()),
            },
            "geometry": {"type": "LineString", "coordinates": coords},
        }],
    }


def build_channel_mapping(chan, lat, lon, dc):
    """
    Mapeo completo canal -> [lon, lat], sin submuestrear, mas el dc.

    Para colocar una deteccion a distancia d (metros):
        canal = round(d / dc)
        indice = canal - channel_min      (los canales son consecutivos)
        coord  = lonlat[indice]
    """
    return {
        "channel_min": int(chan.min()),
        "channel_max": int(chan.max()),
        "dc": dc,  # metros por canal, para convertir metros -> canal
        "lonlat": [[round(float(lon[i]), 6), round(float(lat[i]), 6)]
                   for i in range(len(chan))],
    }


def process(path, outdir, step, dc):
    label = label_from_filename(path)
    chan, lat, lon, dep = load_cable(path)
    print(f"[{label}] canales {chan.min()}-{chan.max()} ({len(chan)} filas), "
          f"prof {dep.max():.0f} a {dep.min():.0f} m")

    geojson = build_geojson(label, chan, lat, lon, step)
    gpath = os.path.join(outdir, f"cable_{label}.geojson")
    with open(gpath, "w") as f:
        json.dump(geojson, f)
    n_line = len(geojson["features"][0]["geometry"]["coordinates"])
    print(f"   -> {gpath}  ({n_line} puntos, step={step})")

    mapping = build_channel_mapping(chan, lat, lon, dc)
    mpath = os.path.join(outdir, f"channel_coords_{label}.json")
    with open(mpath, "w") as f:
        json.dump(mapping, f)
    print(f"   -> {mpath}  (mapeo completo, dc={dc})")


def main():
    ap = argparse.ArgumentParser(description="OOI cable .txt -> GeoJSON + mapeo (uno o varios cables)")
    ap.add_argument("txt", nargs="+", help="uno o varios ficheros *_DAS_latlondepth.txt")
    ap.add_argument("--outdir", default="data", help="carpeta de salida (def: data)")
    ap.add_argument("--step", type=int, default=25, help="submuestreo de la linea (def: 25)")
    ap.add_argument("--dc", type=float, default=2.0417,
                    help="metros por canal (def: 2.0417, medido en los CSV OptaSense)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    for path in args.txt:
        process(path, args.outdir, args.step, args.dc)


if __name__ == "__main__":
    main()