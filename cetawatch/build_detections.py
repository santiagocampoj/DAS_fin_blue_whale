#!/usr/bin/env python3
"""
build_detections.py
-------------------
Lee los CSV de detecciones (*_bbox.csv) de los cables Norte y Sur y los
vuelca a un unico data/detections.json que carga el dashboard CetaWatch.

Cada fila del CSV es una deteccion del modelo. De cada una guardamos:
  - cable        : north / south (deducido del nombre del fichero)
  - cls          : clase del modelo (HF / LF), de la columna ID
  - d0, d1, d_mid: distancia a lo largo del cable en METROS (d_mid = centro)
  - t0, t1, dur  : tiempo dentro del fichero (s) y duracion
  - timestamp_utc: instante absoluto de la deteccion (columna start_datetime_utc)
  - source_file  : nombre del CSV de origen

Por que d_mid (centro) y no el tramo entero: la caja d0-d1 puede abarcar
decenas de km; es la extension de la deteccion en el espectrograma, no la
posicion de la ballena. El centro es la mejor aproximacion simple para el mapa.

Por que metros y no canales: los `di` del CSV Sur son pixeles de imagen
recortada, no canales reales. La distancia en metros es fisica y comun a
los dos cables (ver convert_cable.py / map.js).

Uso:
    python build_detections.py North-C1-...bbox.csv South-C1-...bbox.csv \
        --outdir data
"""

import argparse
import csv
import json
import os


def cable_from_filename(path):
    name = os.path.basename(path).lower()
    if name.startswith("north") or "north" in name:
        return "north"
    if name.startswith("south") or "south" in name:
        return "south"
    return "unknown"


def normalize_timestamp(ts):
    """
    'YYYY-MM-DDTHH:MM:SS.ffffff' -> 'YYYY-MM-DDTHH:MM:SS.fffZ'
    Recorta microsegundos a milisegundos y marca UTC con 'Z', para que
    JavaScript (new Date(...)) lo interprete sin ambiguedad de zona horaria.
    """
    ts = ts.strip()
    if "." in ts:
        head, frac = ts.split(".")
        frac = (frac + "000")[:3]   # 3 decimales (milisegundos)
        ts = f"{head}.{frac}"
    if not ts.endswith("Z"):
        ts += "Z"
    return ts


def parse_csv(path):
    cable = cable_from_filename(path)
    fname = os.path.basename(path)
    out = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            d0 = float(row["d0"])
            d1 = float(row["d1"])
            t0 = float(row["t0"])
            t1 = float(row["t1"])
            det = {
                "id": f"{cable}_{i+1:04d}",
                "cable": cable,
                "cls": row["ID"].strip(),          # HF / LF
                "d0": round(d0, 1),
                "d1": round(d1, 1),
                "d_mid": round((d0 + d1) / 2.0, 1),
                "t0": round(t0, 2),
                "t1": round(t1, 2),
                "dur": round(t1 - t0, 2),
                "timestamp_utc": normalize_timestamp(row["start_datetime_utc"]),
                "source_file": fname,
            }
            out.append(det)
    return out


def main():
    ap = argparse.ArgumentParser(description="bbox CSV -> detections.json")
    ap.add_argument("csv", nargs="+", help="uno o varios *_bbox.csv")
    ap.add_argument("--outdir", default="data", help="carpeta de salida (def: data)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    all_dets = []
    for path in args.csv:
        dets = parse_csv(path)
        cable = dets[0]["cable"] if dets else "?"
        n_hf = sum(1 for d in dets if d["cls"] == "HF")
        n_lf = sum(1 for d in dets if d["cls"] == "LF")
        print(f"[{cable}] {os.path.basename(path)}: {len(dets)} detecciones "
              f"({n_hf} HF, {n_lf} LF)")
        all_dets.extend(dets)

    # Ordenar cronologicamente (util para el 'player' mas adelante)
    all_dets.sort(key=lambda d: d["timestamp_utc"])

    out_path = os.path.join(args.outdir, "detections.json")
    with open(out_path, "w") as f:
        json.dump(all_dets, f, indent=2)
    print(f"\n[OK] {out_path}  ({len(all_dets)} detecciones en total)")


if __name__ == "__main__":
    main()